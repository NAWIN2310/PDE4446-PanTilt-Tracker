"""
Offline Video Replay & Performance Plotter for PDE4446.

Usage:
    python tools/video_replay.py --video path/to/test.mp4 --object Object_A
"""

import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.controller import PanTiltController
from src.detector import ObjectDetector
from src.kalman_tracker import KalmanTracker2D
from src.serial_comm import PanTiltSerialDriver
from src.ui import TrackerHUD


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Video Replay & Metrics Plotter")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--object",
        type=str,
        default="Object_A",
        choices=["Object_A", "Object_B", "Object_C"],
        help="Target profile",
    )
    args = parser.parse_args()

    detector = ObjectDetector()
    detector.set_active_object(args.object)
    kalman = KalmanTracker2D()
    controller = PanTiltController()
    driver = PanTiltSerialDriver()
    hud = TrackerHUD(show_mask_pip=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    times, raw_ex, kf_ex, pan_cmds = [], [], [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detector.detect(frame)
        kf_pos, kf_vel, is_tracking = kalman.step(det.norm_error, det.found, dt)
        pan_cmd, tilt_cmd = controller.compute(kf_pos, is_tracking, dt)
        telemetry = driver.send_command(pan_cmd, tilt_cmd, dt)

        t_sec = frame_idx * dt
        times.append(t_sec)
        raw_ex.append(det.norm_error[0] if det.found else np.nan)
        kf_ex.append(kf_pos[0] if is_tracking else np.nan)
        pan_cmds.append(pan_cmd)
        frame_idx += 1

        active_name = detector.profiles[detector.active_key]["name"]
        display = hud.render(
            frame=frame,
            det=det,
            kf_pos=kf_pos,
            kf_vel=kf_vel,
            is_tracking=is_tracking,
            pan_cmd=pan_cmd,
            tilt_cmd=tilt_cmd,
            telemetry=telemetry,
            active_object_key=detector.active_key,
            active_object_name=active_name,
            fps=fps,
            logging_active=False,
        )
        cv2.imshow("PDE4446 Offline Video Replay", display)
        if (cv2.waitKey(int(dt * 1000)) & 0xFF) in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if times:
        os.makedirs("logs", exist_ok=True)
        out_png = os.path.join("logs", "replay_summary.png")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(times, raw_ex, ".", alpha=0.5, label="Raw HSV Error X")
        ax1.plot(times, kf_ex, "-", linewidth=1.8, label="Kalman Filtered Error X")
        ax1.set_ylabel("Normalized Error X")
        ax1.set_title(f"Offline Replay Performance ({args.object})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(times, pan_cmds, color="tab:orange", label="PID Pan Command [-1, +1]")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Actuator Command")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"[INFO] Saved replay performance plot -> {out_png}")


if __name__ == "__main__":
    main()