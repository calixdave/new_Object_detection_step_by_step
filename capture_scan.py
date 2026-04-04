import cv2
import os
import time

# =========================================================
# Simple Pi-friendly scan capture
# Saves 4 images only: front, right, back, left
# Keys:
#   c = capture current heading
#   q = quit
# =========================================================

SAVE_DIR = "scan_images"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

HEADINGS = ["front", "right", "back", "left"]


def put_text(img, text, y, scale=0.7, thickness=2):
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Set modest resolution for Pi stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Camera opened.")
    print("Instructions:")
    print("  Press 'c' to capture current heading")
    print("  Press 'q' to quit")
    print("Capture order: front -> right -> back -> left")

    idx = 0
    last_capture_msg = ""

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR: Failed to read frame from camera.")
            break

        display = frame.copy()

        if idx < len(HEADINGS):
            current_heading = HEADINGS[idx]
            put_text(display, f"Current heading: {current_heading}", 30, 0.9, 2)
            put_text(display, "Press 'c' to capture this view", 65)
            put_text(display, "Rotate camera/robot manually before each capture", 95)
            put_text(display, "Press 'q' to quit", 125)
        else:
            put_text(display, "All 4 captures completed.", 30, 0.9, 2)
            put_text(display, "Press 'q' to quit", 65)

        if last_capture_msg:
            put_text(display, last_capture_msg, FRAME_HEIGHT - 20, 0.65, 2)

        cv2.imshow("capture_scan", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit requested.")
            break

        elif key == ord('c'):
            if idx >= len(HEADINGS):
                print("All headings already captured.")
                last_capture_msg = "All headings already captured"
                continue

            heading = HEADINGS[idx]
            filename = os.path.join(SAVE_DIR, f"{heading}.jpg")

            ok = cv2.imwrite(filename, frame)
            if ok:
                print(f"Saved: {filename}")
                last_capture_msg = f"Saved {heading}.jpg"
                idx += 1
                time.sleep(0.4)  # small pause to avoid double-capture
            else:
                print(f"ERROR: Failed to save {filename}")
                last_capture_msg = f"Failed to save {heading}.jpg"

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")


if __name__ == "__main__":
    main()
