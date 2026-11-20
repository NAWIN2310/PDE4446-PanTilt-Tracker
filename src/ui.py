"""
OpenCV HUD & Visualization Module for PDE4446 Pan-Tilt Tracker.

Renders:
- Center crosshair and deadband target lock ring
- Raw HSV detection contour / rotated bounding box (Object_C)
- Kalman-filtered position and velocity vector
- Real-time telemetry panel (FPS, active object, normalized error, PID output,
  servo angles, serial status, and CSV logging indicator)
"""

from typing import Tuple
import cv2
import numpy as np

from src.detector import DetectionResult
from src.serial_comm import RigTelemetry


class TrackerHUD:
    """Draws real-time visual feedback and telemetry onto the camera frame."""

    def __init__(self, show_mask_pip: bool = True) -> None:
        self.show_mask_pip = show_mask_pip

    def toggle_mask_pip(self) -> None:
        self.show_mask_pip = not self.show_mask_pip

    def render(
        self,
        frame: np.ndarray,
        det: DetectionResult,
        kf_pos: Tuple[float, float],
        kf_vel: Tuple[float, float],
        is_tracking: bool,
        pan_cmd: float,
        tilt_cmd: float,
        telemetry: RigTelemetry,
        active_object_key: str,
        active_object_name: str,
        fps: float,
        logging_active: bool,
        deadband: float = 0.025,
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        cx_mid, cy_mid = w // 2, h // 2

        # 1. Center crosshair & deadband lock zone
        db_rx = max(8, int(deadband * cx_mid))
        db_ry = max(8, int(deadband * cy_mid))
        cv2.line(canvas, (cx_mid - 25, cy_mid), (cx_mid + 25, cy_mid), (200, 200, 200), 1)
        cv2.line(canvas, (cx_mid, cy_mid - 25), (cx_mid, cy_mid + 25), (200, 200, 200), 1)
        cv2.ellipse(canvas, (cx_mid, cy_mid), (db_rx, db_ry), 0, 0, 360, (140, 140, 140), 1)

        # 2. Raw detection overlay
        if det.found and det.center_px is not None:
            raw_x, raw_y = det.center_px
            if det.rotated_box is not None:
                cv2.drawContours(canvas, [det.rotated_box], 0, (0, 215, 255), 2)
                cv2.putText(
                    canvas,
                    f"Angle: {det.angle_deg:.1f} deg",
                    (raw_x + 12, raw_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 215, 255),
                    1,
                )
            elif det.bbox is not None:
                bx, by, bw, bh = det.bbox
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            cv2.circle(canvas, (raw_x, raw_y), 5, (0, 255, 0), -1)

        # 3. Kalman-filtered target & velocity arrow
        if is_tracking:
            kf_px = int(cx_mid + kf_pos[0] * cx_mid)
            kf_py = int(cy_mid + kf_pos[1] * cy_mid)
            color = (0, 255, 255) if det.found else (0, 140, 255)
            status_txt = "LOCKED" if det.found else "COASTING (KF)"

            cv2.circle(canvas, (kf_px, kf_py), 9, color, 2)
            cv2.line(canvas, (cx_mid, cy_mid), (kf_px, kf_py), color, 1)

            # Velocity vector projection (0.15s ahead)
            pred_px = int(kf_px + kf_vel[0] * cx_mid * 0.15)
            pred_py = int(kf_py + kf_vel[1] * cy_mid * 0.15)
            cv2.arrowedLine(canvas, (kf_px, kf_py), (pred_px, pred_py), (255, 100, 255), 2)
        else:
            status_txt = "SEARCHING"
            color = (0, 80, 255)

        # 4. Semi-transparent telemetry banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (430, 152), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        lines = [
            f"Target: {active_object_key} ({active_object_name}) | FPS: {fps:.1f}",
            f"State: {status_txt} | Err(X,Y): ({kf_pos[0]:+.3f}, {kf_pos[1]:+.3f})",
            f"PID Cmd (Pan,Tilt): ({pan_cmd:+.3f}, {tilt_cmd:+.3f})",
            f"Rig Angles: Pan={telemetry.pan_deg:+.1f} deg  Tilt={telemetry.tilt_deg:+.1f} deg",
            f"Serial: {telemetry.mode_label} | Packet: {telemetry.last_packet}",
            f"Log: {'REC (CSV)' if logging_active else 'OFF'} | [1/2/3] Obj [L] Log [M] Mask [C] Center [Q] Quit",
        ]
        for idx, text in enumerate(lines):
            txt_color = color if idx == 1 else (235, 235, 235)
            cv2.putText(
                canvas,
                text,
                (16, 28 + idx * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                txt_color,
                1,
                cv2.LINE_AA,
            )

        # 5. Picture-in-Picture Binary Mask (bottom-right)
        if self.show_mask_pip and det.mask is not None:
            pip_w, pip_h = 180, 135
            mask_small = cv2.resize(det.mask, (pip_w, pip_h))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_bgr, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 0), 1)
            cv2.putText(
                mask_bgr,
                "HSV Mask",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )
            canvas[h - pip_h - 10 : h - 10, w - pip_w - 10 : w - 10] = mask_bgr

        return canvas