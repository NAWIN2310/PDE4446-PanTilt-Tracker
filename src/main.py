"""
Main Execution Entry Point for PDE4446 Pan-Tilt Visual Tracker.

Usage:
    python -m src.main                  # Live webcam + simulation mode
    python -m src.main --camera 1       # Use external USB webcam (index 1)
    python -m src.main --port COM3      # Connect to physical Arduino/servo rig
    python -m src.main --video path.mp4 # Offline replay from recorded video

Keyboard Controls:
    1 / 2 / 3 : Switch between Object_A, Object_B, and Object_C
    L         : Start / Stop CSV telemetry logging into logs/
    M         : Toggle Picture-in-Picture binary mask
    C         : Re-center pan/tilt rig and reset Kalman/PID states
    Q / ESC   : Quit cleanly
"""

import argparse
import csv
from datetime import datetime
import os
import time
from typing import Optional
import cv2

from src.controller import PanTiltController
from src.detector import ObjectDetector
from src.kalman_tracker import KalmanTracker2D
from src.serial_comm import PanTiltSerialDriver
from src.ui import TrackerHUD


class TelemetryLogger:
    """Records frame-by-frame tracking error, PID commands, and rig angles to CSV."""

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.file = None
        self.writer = None
        self.filepath: Optional[str] = None
        self.start_time: float = 0.0

    @property
    def active(self) -> bool:
        return self.file is not None

    def toggle(self) -> None:
        if self.active:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self.active:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.log_dir, f"tracking_log_{stamp}.csv")
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "elapsed_s",
                "object_key",
                "detected",
                "raw_err_x",
                "raw_err_y",
                "kf_err_x",
                "kf_err_y",
                "kf_vel_x",
                "kf_vel_y",
                "pan_cmd",
                "tilt_cmd",
                "pan_deg",
                "tilt_deg",
                "fps",
            ]
        )
        self.start_time = time.perf_counter()
        print(f"[INFO] Started CSV logging -> {self.filepath}")

    def log_row(
        self,
        object_key: str,
        detected: bool,
        raw_err: tuple,
        kf_err: tuple,
        kf_vel: tuple,
        pan_cmd: float,
        tilt_cmd: float,
        pan_deg: float,
        tilt_deg: float,
        fps: float,
    ) -> None:
        if not self.active or self.writer is None:
            return
        elapsed = time.perf_counter() - self.start_time
        self.writer.writerow(
            [
                f"{elapsed:.4f}",
                object_key,
                int(detected),
                f"{raw_err[0]:.4f}",
                f"{raw_err[1]:.4f}",
                f"{kf_err[0]:.4f}",
                f"{kf_err[1]:.4f}",
                f"{kf_vel[0]:.4f}",
                f"{kf_vel[1]:.4f}",
                f"{pan_cmd:.4f}",
                f"{tilt_cmd:.4f}",
                f"{pan_deg:.2f}",
                f"{tilt_deg:.2f}",
                f"{fps:.1f}",
            ]
        )

    def stop(self) -> None:
        if self.file is not None:
            self.file.close()
            print(f"[INFO] Saved CSV log -> {self.filepath}")
        self.file = None
        self.writer = None
        self.filepath = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDE4446 Pan-Tilt Visual Tracker")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional video file path")
    parser.add_argument("--port", type=str, default=None, help="Serial port e.g. COM3")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--object",
        type=str,
        default="Object_A",
        choices=["Object_A", "Object_B", "Object_C"],
        help="Initial target profile",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = ObjectDetector()
    detector.set_active_object(args.object)

    kalman = KalmanTracker2D()
    controller = PanTiltController()
    driver = PanTiltSerialDriver(port=args.port, baudrate=args.baud)
    hud = TrackerHUD(show_mask_pip=True)
    logger = TelemetryLogger()

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    prev_time = time.perf_counter()
    fps_smooth = 30.0

    print("[INFO] Pan-Tilt Tracker running. Press 'Q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    # Loop video replay seamlessly
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            now = time.perf_counter()
            dt = max(1e-3, now - prev_time)
            prev_time = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # 1. Detect target in HSV space
            det = detector.detect(frame)

            # 2. Kalman filtering & coasting prediction
            kf_pos, kf_vel, is_tracking = kalman.step(
                measured_pos=det.norm_error,
                found=det.found,
                dt=dt,
            )

            # 3. PID control commands [-1.0, +1.0]
            pan_cmd, tilt_cmd = controller.compute(
                error_xy=kf_pos,
                is_tracking=is_tracking,
                dt=dt,
            )

            # 4. Actuate serial hardware or simulated rig
            telemetry = driver.send_command(pan_cmd=pan_cmd, tilt_cmd=tilt_cmd, dt=dt)

            # 5. Optional CSV telemetry logging
            logger.log_row(
                object_key=detector.active_key,
                detected=det.found,
                raw_err=det.norm_error,
                kf_err=kf_pos,
                kf_vel=kf_vel,
                pan_cmd=pan_cmd,
                tilt_cmd=tilt_cmd,
                pan_deg=telemetry.pan_deg,
                tilt_deg=telemetry.tilt_deg,
                fps=fps_smooth,
            )

            # 6. Render HUD
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
                fps=fps_smooth,
                logging_active=logger.active,
                deadband=controller.pan_pid.deadband,
            )
            cv2.imshow("PDE4446 Pan-Tilt Visual Tracker", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("1"):
                detector.set_active_object("Object_A")
                kalman.reset()
                controller.reset()
            elif key == ord("2"):
                detector.set_active_object("Object_B")
                kalman.reset()
                controller.reset()
            elif key == ord("3"):
                detector.set_active_object("Object_C")
                kalman.reset()
                controller.reset()
            elif key == ord("l"):
                logger.toggle()
            elif key == ord("m"):
                hud.toggle_mask_pip()
            elif key == ord("c"):
                kalman.reset()
                controller.reset()
                driver.center_rig()
    finally:
        logger.stop()
        driver.center_rig()
        driver.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()