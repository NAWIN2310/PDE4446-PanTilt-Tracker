"""
Interactive HSV & Shape Calibration Tool for PDE4446 Pan-Tilt Tracker.

Usage:
    python tools/hsv_tuner.py

Controls:
    - Adjust trackbars (H/S/V min & max, Min Area, Min Circularity) while holding
      the target object in front of the webcam.
    - Press '1', '2', or '3' to load presets for Object_A, Object_B, or Object_C.
    - Press 's' to save current slider values to data/objects.json for the active object.
    - Press 'q' or ESC to quit.
"""

import json
import os
import cv2
import numpy as np

CONFIG_PATH = os.path.join("data", "objects.json")
OBJECT_KEYS = ["Object_A", "Object_B", "Object_C"]


def nothing(_val: int) -> None:
    """No-op callback required by OpenCV createTrackbar."""


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved updated calibration to {path}")


def set_trackbars_from_profile(window_name: str, profile: dict) -> None:
    low = profile["hsv_lower"]
    high = profile["hsv_upper"]
    cv2.setTrackbarPos("H Min", window_name, low[0])
    cv2.setTrackbarPos("S Min", window_name, low[1])
    cv2.setTrackbarPos("V Min", window_name, low[2])
    cv2.setTrackbarPos("H Max", window_name, high[0])
    cv2.setTrackbarPos("S Max", window_name, high[1])
    cv2.setTrackbarPos("V Max", window_name, high[2])
    cv2.setTrackbarPos("Min Area", window_name, int(profile.get("min_area", 400)))
    cv2.setTrackbarPos(
        "Min Circ x100", window_name, int(profile.get("min_circularity", 0.5) * 100)
    )


def main() -> None:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Run from repository root.")

    config = load_config(CONFIG_PATH)
    active_idx = 0
    active_key = OBJECT_KEYS[active_idx]

    win_ctrl = "HSV & Shape Controls"
    cv2.namedWindow(win_ctrl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_ctrl, 420, 360)

    cv2.createTrackbar("H Min", win_ctrl, 0, 179, nothing)
    cv2.createTrackbar("S Min", win_ctrl, 0, 255, nothing)
    cv2.createTrackbar("V Min", win_ctrl, 0, 255, nothing)
    cv2.createTrackbar("H Max", win_ctrl, 179, 179, nothing)
    cv2.createTrackbar("S Max", win_ctrl, 255, 255, nothing)
    cv2.createTrackbar("V Max", win_ctrl, 255, 255, nothing)
    cv2.createTrackbar("Min Area", win_ctrl, 400, 10000, nothing)
    cv2.createTrackbar("Min Circ x100", win_ctrl, 50, 100, nothing)

    set_trackbars_from_profile(win_ctrl, config[active_key])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_min = cv2.getTrackbarPos("H Min", win_ctrl)
        s_min = cv2.getTrackbarPos("S Min", win_ctrl)
        v_min = cv2.getTrackbarPos("V Min", win_ctrl)
        h_max = cv2.getTrackbarPos("H Max", win_ctrl)
        s_max = cv2.getTrackbarPos("S Max", win_ctrl)
        v_max = cv2.getTrackbarPos("V Max", win_ctrl)
        min_area = max(50, cv2.getTrackbarPos("Min Area", win_ctrl))
        min_circ = cv2.getTrackbarPos("Min Circ x100", win_ctrl) / 100.0

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preview = frame.copy()

        best_contour = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity >= min_circ and area > best_area:
                best_area = area
                best_contour = cnt

        if best_contour is not None:
            moments = cv2.moments(best_contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                cv2.drawContours(preview, [best_contour], -1, (0, 255, 0), 2)
                cv2.circle(preview, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(
                    preview,
                    f"Area: {int(best_area)}",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        label = f"Editing: {active_key} ({config[active_key]['name']}) | [1/2/3] Switch | [S] Save | [Q] Quit"
        cv2.putText(preview, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((preview, mask_bgr))
        cv2.imshow("HSV Tuner (Left: Detection | Right: Binary Mask)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            active_idx = int(chr(key)) - 1
            active_key = OBJECT_KEYS[active_idx]
            set_trackbars_from_profile(win_ctrl, config[active_key])
            print(f"[INFO] Switched to {active_key}")
        elif key == ord("s"):
            config[active_key]["hsv_lower"] = [h_min, s_min, v_min]
            config[active_key]["hsv_upper"] = [h_max, s_max, v_max]
            config[active_key]["min_area"] = int(min_area)
            config[active_key]["min_circularity"] = round(min_circ, 2)
            save_config(CONFIG_PATH, config)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()