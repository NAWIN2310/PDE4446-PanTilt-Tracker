"""
2D Constant-Velocity Kalman Filter for PDE4446 Pan-Tilt Tracker.

State vector: x = [px, py, vx, vy]^T in normalized frame coordinates [-1.0, +1.0].
Measurement vector: z = [px, py]^T from ObjectDetector.
"""

from typing import Tuple
import numpy as np


class KalmanTracker2D:
    """Smooths target position and predicts motion across missed detections."""

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 5e-3,
        max_coast_frames: int = 8,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_coast_frames = max_coast_frames

        self.initialized: bool = False
        self.missed_frames: int = 0

        # State [px, py, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float64)
        # Covariance matrix
        self.P = np.eye(4, dtype=np.float64)
        # Measurement matrix (observes px, py)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.R = np.eye(2, dtype=np.float64) * self.measurement_noise
        self.I = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        """Reset filter state when target is lost for too long."""
        self.initialized = False
        self.missed_frames = 0
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def _transition_matrix(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _process_covariance(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q = self.process_noise
        return q * np.array(
            [
                [dt4, 0.0, dt3, 0.0],
                [0.0, dt4, 0.0, dt3],
                [dt3, 0.0, dt2, 0.0],
                [dt3, 0.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    def step(
        self,
        measured_pos: Tuple[float, float],
        found: bool,
        dt: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], bool]:
        """
        Advance filter by one time step.

        Returns:
            (filtered_pos, estimated_vel, is_tracking)
        """
        dt = max(1e-3, min(0.2, float(dt)))

        if not self.initialized:
            if not found:
                return (0.0, 0.0), (0.0, 0.0), False
            self.x = np.array(
                [[measured_pos[0]], [measured_pos[1]], [0.0], [0.0]], dtype=np.float64
            )
            self.P = np.eye(4, dtype=np.float64) * 0.1
            self.initialized = True
            self.missed_frames = 0
            return (float(self.x[0, 0]), float(self.x[1, 0])), (0.0, 0.0), True

        # 1. Predict
        F = self._transition_matrix(dt)
        Q = self._process_covariance(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # 2. Update if measurement is available
        if found:
            self.missed_frames = 0
            z = np.array([[measured_pos[0]], [measured_pos[1]]], dtype=np.float64)
            y = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + (K @ y)
            self.P = (self.I - (K @ self.H)) @ self.P
        else:
            self.missed_frames += 1
            self.x[2, 0] *= 0.90
            self.x[3, 0] *= 0.90
            if self.missed_frames > self.max_coast_frames:
                self.reset()
                return (0.0, 0.0), (0.0, 0.0), False

        px = float(np.clip(self.x[0, 0], -1.0, 1.0))
        py = float(np.clip(self.x[1, 0], -1.0, 1.0))
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        return (px, py), (vx, vy), True