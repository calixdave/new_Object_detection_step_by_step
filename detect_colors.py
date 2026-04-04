import os
import json
import cv2
import joblib
import numpy as np

# =========================================================
# detect_colors.py
# Pi-friendly color detection from saved scan images
# =========================================================

MODEL_PATH = "tile_color_model.joblib"
SCAN_DIR = "scan_images"
HEADINGS = ["front", "right", "back", "left"]

ROI_TOP_FRAC = 0.62
ROI_BOT_FRAC = 0.93

SLOT_PAD_X_FRAC = 0.06
SLOT_PAD_Y_FRAC = 0.10

CONF_THRESH = 0.40

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


def load_classifier(model_path):
    obj = joblib.load(model_path)

    # If the joblib file already is the model, use it directly
    if hasattr(obj, "predict") or hasattr(obj, "predict_proba"):
        print("Loaded model directly from joblib file.")
        return obj

    # If wrapped inside a dict, try common keys
    if isinstance(obj, dict):
        print("Joblib file contains a dict. Keys found:", list(obj.keys()))

        candidate_keys = [
            "model", "clf", "classifier", "svc", "svm", "estimator"
        ]

        for key in candidate_keys:
            if key in obj:
                candidate = obj[key]
                if hasattr(candidate, "predict") or hasattr(candidate, "predict_proba"):
                    print(f"Using classifier from dict key: '{key}'")
                    return candidate

        # As a fallback, scan dict values
        for key, val in obj.items():
            if hasattr(val, "predict") or hasattr(val, "predict_proba"):
                print(f"Using classifier found in dict value under key: '{key}'")
                return val

    raise ValueError(
        "Could not find a classifier inside tile_color_model.joblib. "
        "The file loaded, but no usable model object was found."
    )


def extract_features(img):
    h, w = img.shape[:2]
    y0, y1 = int(0.25 * h), int(0.75 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)

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


def classify_tile(model, tile_bgr):
    feats = extract_features(tile_bgr)
    if feats is None:
        return "unknown", 0.0, "?"

    x = feats.reshape(1, -1)

    # Try predict_proba first
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = model.classes_
        best_idx = int(np.argmax(probs))
        label = str(classes[best_idx]).lower()
        conf = float(probs[best_idx])
    else:
        label = str(model.predict(x)[0]).lower()
        conf = 1.0

    if conf < CONF_THRESH or label not in LABEL_TO_CHAR:
        return "unknown", conf, "?"

    return label, conf, LABEL_TO_CHAR[label]


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
            vals.append(mat[(col, row)])
        print(" ".join(vals))


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return

    try:
        model = load_classifier(MODEL_PATH)
    except Exception as e:
        print("ERROR loading classifier:", e)
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

        for i, tile in enumerate(slots):
            label, conf, ch = classify_tile(model, tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "label": label,
                "confidence": round(conf, 4),
                "char": ch
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
    print("Done.")


if __name__ == "__main__":
    main()
