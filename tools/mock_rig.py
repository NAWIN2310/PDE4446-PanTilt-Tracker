"""
Offline Closed-Loop Pan-Tilt Simulator for PDE4446.

Usage:
    python tools/mock_rig.py
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.controller import PanTiltController
from src.kalman_tracker import KalmanTracker2D
from src.serial_comm import PanTiltSerialDriver


def main() -> None:
    width, height = 800, 600
    cx_mid, cy_mid = width // 2, height // 2

    kalman = KalmanTracker2D()
    controller = PanTiltController()
    driver = PanTiltSerialDriver()

    t_start = time.perf_counter()
    prev_time = t_start
    occlude = False

    print("[INFO] Offline Mock Rig running. [O] Toggle Occlusion | [C] Center | [Q] Quit")

    while True:
        now = time.perf_counter()
        dt = max(1e-3, now - prev_time)
        prev_time = now
        elapsed = now - t_start

        target_x = 0.55 * np.sin(0.9 * elapsed)
        target_y = 0.40 * np.sin(1.8 * elapsed)

        cam_err_x = float(np.clip(target_x - (driver.pan_deg / 90.0), -1.0, 1.0))
        cam_err_y = float(np.clip(target_y - (driver.tilt_deg / 45.0), -1.0, 1.0))

        found = not occlude
        if found:
            meas_x = cam_err_x + float(np.random.normal(0.0, 0.008))
            meas_y = cam_err_y + float(np.random.normal(0.0, 0.008))
        else:
            meas_x, meas_y = 0.0, 0.0

        kf_pos, kf_vel, is_tracking = kalman.step((meas_x, meas_y), found=found, dt=dt)
        pan_cmd, tilt_cmd = controller.compute(kf_pos, is_tracking=is_tracking, dt=dt)
        telemetry = driver.send_command(pan_cmd, tilt_cmd, dt=dt)

        canvas = np.full((height, width, 3), (24, 26, 32), dtype=np.uint8)
        cv2.line(canvas, (cx_mid - 30, cy_mid), (cx_mid + 30, cy_mid), (120, 120, 120), 1)
        cv2.line(canvas, (cx_mid, cy_mid - 30), (cx_mid, cy_mid + 30), (120, 120, 120), 1)

        true_px = int(cx_mid + cam_err_x * cx_mid)
        true_py = int(cy_mid + cam_err_y * cy_mid)
        if found:
            cv2.circle(canvas, (true_px, true_py), 16, (0, 165, 255), -1)
        else:
            cv2.circle(canvas, (true_px, true_py), 16, (80, 80, 90), 1)

        if is_tracking:
            kf_px = int(cx_mid + kf_pos[0] * cx_mid)
            kf_py = int(cy_mid + kf_pos[1] * cy_mid)
            color = (0, 255, 255) if found else (0, 140, 255)
            cv2.circle(canvas, (kf_px, kf_py), 22, color, 2)
            cv2.line(canvas, (cx_mid, cy_mid), (kf_px, kf_py), color, 1)

        info = [
            "OFFLINE CLOSED-LOOP MOCK RIG SIMULATOR",
            f"Target Visible: {'YES' if found else 'OCCLUDED (KF Coasting)'}",
            f"KF Error (X, Y): ({kf_pos[0]:+.3f}, {kf_pos[1]:+.3f})",
            f"PID Cmd (Pan, Tilt): ({pan_cmd:+.3f}, {tilt_cmd:+.3f})",
            f"Rig Angles: Pan={telemetry.pan_deg:+.1f} deg | Tilt={telemetry.tilt_deg:+.1f} deg",
            f"Serial Frame: {telemetry.last_packet}",
            "Controls: [O] Toggle Occlusion  [C] Center Rig  [Q] Quit",
        ]
        for idx, line in enumerate(info):
            cv2.putText(
                canvas,
                line,
                (18, 30 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("PDE4446 Offline Mock Rig Simulator", canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("o"):
            occlude = not occlude
        elif key == ord("c"):
            kalman.reset()
            controller.reset()
            driver.center_rig()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()