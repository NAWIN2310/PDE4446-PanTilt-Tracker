"""
Normalized 2-Axis PID Controller for PDE4446 Pan-Tilt Tracker.

Computes normalized actuator commands u_pan, u_tilt in [-1.0, +1.0] from
normalized visual tracking error e_x, e_y in [-1.0, +1.0].
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float


class AxisPID:
    """Single-axis PID controller with deadband, anti-windup, and low-pass D-term."""

    def __init__(
        self,
        gains: PIDGains,
        deadband: float = 0.025,
        integral_limit: float = 0.4,
        output_limit: float = 1.0,
        d_filter_alpha: float = 0.35,
    ) -> None:
        self.gains = gains
        self.deadband = deadband
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.d_filter_alpha = d_filter_alpha

        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.filtered_derivative: float = 0.0
        self._first_step: bool = True

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self._first_step = True

    def update(self, error: float, dt: float) -> float:
        dt = max(1e-3, min(0.2, float(dt)))

        if abs(error) < self.deadband:
            error = 0.0
            self.integral *= 0.92

        p_term = self.gains.kp * error

        self.integral = float(
            np.clip(
                self.integral + error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        i_term = self.gains.ki * self.integral

        if self._first_step:
            raw_d = 0.0
            self._first_step = False
        else:
            raw_d = (error - self.prev_error) / dt

        self.filtered_derivative = (
            self.d_filter_alpha * raw_d
            + (1.0 - self.d_filter_alpha) * self.filtered_derivative
        )
        d_term = self.gains.kd * self.filtered_derivative
        self.prev_error = error

        output = p_term + i_term + d_term
        return float(np.clip(output, -self.output_limit, self.output_limit))


class PanTiltController:
    """2-DOF Pan/Tilt visual servoing controller."""

    def __init__(
        self,
        pan_gains: PIDGains = PIDGains(kp=0.65, ki=0.05, kd=0.08),
        tilt_gains: PIDGains = PIDGains(kp=0.65, ki=0.05, kd=0.08),
        deadband: float = 0.025,
        invert_pan: bool = False,
        invert_tilt: bool = False,
    ) -> None:
        self.pan_pid = AxisPID(gains=pan_gains, deadband=deadband)
        self.tilt_pid = AxisPID(gains=tilt_gains, deadband=deadband)
        self.invert_pan = invert_pan
        self.invert_tilt = invert_tilt

        self.pan_cmd: float = 0.0
        self.tilt_cmd: float = 0.0

    def reset(self) -> None:
        self.pan_pid.reset()
        self.tilt_pid.reset()
        self.pan_cmd = 0.0
        self.tilt_cmd = 0.0

    def set_gains(
        self,
        kp_pan: float,
        ki_pan: float,
        kd_pan: float,
        kp_tilt: float,
        ki_tilt: float,
        kd_tilt: float,
    ) -> None:
        self.pan_pid.gains = PIDGains(kp=kp_pan, ki=ki_pan, kd=kd_pan)
        self.tilt_pid.gains = PIDGains(kp=kp_tilt, ki=ki_tilt, kd=kd_tilt)

    def compute(
        self, error_xy: Tuple[float, float], is_tracking: bool, dt: float
    ) -> Tuple[float, float]:
        """Compute normalized (pan, tilt) commands in [-1.0, +1.0]."""
        if not is_tracking:
            self.reset()
            return 0.0, 0.0

        err_x, err_y = error_xy
        if self.invert_pan:
            err_x = -err_x
        if self.invert_tilt:
            err_y = -err_y

        self.pan_cmd = self.pan_pid.update(err_x, dt)
        self.tilt_cmd = self.tilt_pid.update(err_y, dt)
        return self.pan_cmd, self.tilt_cmd