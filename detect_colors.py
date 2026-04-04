import os
import json
import cv2
import joblib
import numpy as np

MODEL_PATH = "tile_color_model.joblib"
SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_tiles"
HEADINGS = ["front", "right", "back", "left"]

ROI_TOP_FRAC = 0.55
ROI_BOT_FRAC = 0.95

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

CONF_THRESH = 0.00

LABEL_TO_CHAR = {
    "blue": "B",
    "green": "G",
    "red": "R",
    "yellow": "Y",
    "pink": "M",
    "purple": "P"
}

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


def load_model_bundle(model_path):
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        print("Joblib file contains a dict. Keys found:", list(obj.keys()))

        if "model" not in obj:
            raise ValueError("Joblib dict does not contain key 'model'.")

        model = obj["model"]
        class_names = obj.get("classes", None)

        if class_names is not None:
            class_names = [str(x).lower() for x in class_names]
            print("Loaded class names from joblib:", class_names)
        else:
            print("No explicit class names found in joblib.")

        return model, class_names

    # fallback if file is directly a model
    if hasattr(obj, "predict") or hasattr(obj, "predict_proba"):
        print("Loaded model directly from joblib file.")
        return obj, None

    raise ValueError("No usable classifier found in joblib file.")


def extract_features(img):
    h, w = img.shape[:2]

    y0 = int(0.25 * h)
    y1 = int(0.75 * h)
    x0 = int(0.25 * w)
    x1 = int(0.75 * w)

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    feats = []
    for arr in (lab, hsv):
        flat = arr.reshape(-1, 3).astype(np.float32)
        feats.extend(flat.mean(axis=0).tolist())
        feats.extend(flat.std(axis=0).tolist())

    return np.array(feats, dtype=np.float32)


def normalize_predicted_label(raw_label, class_names):
    """
    Convert model output to a usable color string.
    Handles:
      - direct string labels like 'blue'
      - numeric labels like 0,1,2 using class_names list
    """
    s = str(raw_label).lower()

    # already a direct color name
    if s in LABEL_TO_CHAR:
        return s

    # numeric index -> class_names
    if class_names is not None:
        try:
            idx = int(raw_label)
            if 0 <= idx < len(class_names):
                mapped = str(class_names[idx]).lower()
                return mapped
        except Exception:
            pass

    return s


def classify_tile(model, class_names, tile_bgr):
    feats = extract_features(tile_bgr)
    if feats is None:
        return "unknown", 0.0, "?", {}

    x = feats.reshape(1, -1)
    prob_map = {}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]

        if hasattr(model, "classes_"):
            raw_classes = list(model.classes_)
        else:
            raw_classes = list(range(len(probs)))

        best_idx = int(np.argmax(probs))
        raw_label = raw_classes[best_idx]
        label = normalize_predicted_label(raw_label, class_names)
        conf = float(probs[best_idx])

        for c, p in zip(raw_classes, probs):
            cname = normalize_predicted_label(c, class_names)
            prob_map[str(cname)] = float(p)
    else:
        raw_label = model.predict(x)[0]
        label = normalize_predicted_label(raw_label, class_names)
        conf = 1.0

    ch = LABEL_TO_CHAR[label] if label in LABEL_TO_CHAR else "?"

    if conf < CONF_THRESH:
        return "unknown", conf, "?", prob_map

    return label, conf, ch, prob_map


def get_three_slot_rois(img):
    h, w = img.shape[:2]

    y0 = int(ROI_TOP_FRAC * h)
    y1 = int(ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]

    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def pretty_print_matrix(mat):
    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(mat.get((col, row), "?"))
        print(" ".join(vals))


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return

    try:
        model, class_names = load_model_bundle(MODEL_PATH)
    except Exception as e:
        print("ERROR loading model:", e)
        return

    final_grid = {
        (-1, +1): "?",
        ( 0, +1): "?",
        (+1, +1): "?",
        (-1,  0): "?",
        ( 0,  0): "A",
        (+1,  0): "?",
        (-1, -1): "?",
        ( 0, -1): "?",
        (+1, -1): "?",
    }

    detailed = {}

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            print(f"ERROR: Missing image: {path}")
            return

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        slots = get_three_slot_rois(img)
        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading}")
            return

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            dbg_name = os.path.join(DEBUG_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, tile)

            label, conf, ch, prob_map = classify_tile(model, class_names, tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            print(f"  slot {i}: label={label}, conf={conf:.4f}, char={ch}, saved={dbg_name}")
            if prob_map:
                rounded_probs = {k: round(v, 4) for k, v in prob_map.items()}
                print("    probs:", rounded_probs)

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "label": label,
                "confidence": round(conf, 4),
                "char": ch,
                "debug_crop": dbg_name,
                "probs": {k: round(v, 4) for k, v in prob_map.items()}
            })

        detailed[heading] = heading_info

    print("\nFinal 3x3 color matrix:")
    pretty_print_matrix(final_grid)

    out = {
        "center": [0, 0],
        "agent": "A",
        "grid_letters": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    with open("color_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved: color_results.json")
    print(f"Saved debug crops in: {DEBUG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
