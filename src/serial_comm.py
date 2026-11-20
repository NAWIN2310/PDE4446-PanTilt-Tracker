"""
Serial Communication & Simulated Pan-Tilt Rig Driver for PDE4446.

Framing protocol sent to microcontroller (115200 baud):
    <PAN:+0.245,TILT:-0.110>\n

If no physical serial port is connected (or simulation mode is selected),
an internal 2-DOF kinematic rig model simulates pan/tilt servo angles
in degrees so the entire closed-loop pipeline can be tested and logged offline.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


@dataclass
class RigTelemetry:
    """Current actuator angles and communication status."""

    pan_deg: float
    tilt_deg: float
    connected: bool
    mode_label: str
    last_packet: str


class PanTiltSerialDriver:
    """Manages real USB-Serial communication or simulated pan-tilt dynamics."""

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        pan_limits: Tuple[float, float] = (-90.0, 90.0),
        tilt_limits: Tuple[float, float] = (-45.0, 45.0),
        max_rate_deg_s: float = 120.0,
    ) -> None:
        self.baudrate = baudrate
        self.pan_limits = pan_limits
        self.tilt_limits = tilt_limits
        self.max_rate_deg_s = max_rate_deg_s

        self.ser = None
        self.port_name: Optional[str] = None
        self.pan_deg: float = 0.0
        self.tilt_deg: float = 0.0
        self.last_packet: str = "<PAN:+0.000,TILT:+0.000>"

        if port:
            self.connect(port)

    @staticmethod
    def list_available_ports() -> List[str]:
        """Return list of detected COM/tty serial port names."""
        if serial is None:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port: str) -> bool:
        """Attempt to open a physical serial port; falls back cleanly on error."""
        self.disconnect()
        if serial is None:
            return False
        try:
            self.ser = serial.Serial(port=port, baudrate=self.baudrate, timeout=0.02)
            self.port_name = port
            return True
        except Exception as exc:
            print(f"[WARN] Could not open serial port {port}: {exc}. Using SIM mode.")
            self.ser = None
            self.port_name = None
            return False

    def disconnect(self) -> None:
        """Close active serial connection if open."""
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.port_name = None

    def center_rig(self) -> None:
        """Reset pan/tilt angles to neutral (0, 0) degrees."""
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self.send_command(0.0, 0.0, dt=0.033)

    def send_command(
        self, pan_cmd: float, tilt_cmd: float, dt: float
    ) -> RigTelemetry:
        """
        Send normalized commands [-1.0, +1.0] to hardware or update sim rig.
        """
        dt = max(1e-3, min(0.2, float(dt)))
        pan_cmd = float(np.clip(pan_cmd, -1.0, 1.0))
        tilt_cmd = float(np.clip(tilt_cmd, -1.0, 1.0))

        # Integrate angular velocity into bounded servo angles
        self.pan_deg = float(
            np.clip(
                self.pan_deg + pan_cmd * self.max_rate_deg_s * dt,
                self.pan_limits[0],
                self.pan_limits[1],
            )
        )
        self.tilt_deg = float(
            np.clip(
                self.tilt_deg + tilt_cmd * self.max_rate_deg_s * dt,
                self.tilt_limits[0],
                self.tilt_limits[1],
            )
        )

        packet = f"<PAN:{pan_cmd:+.3f},TILT:{tilt_cmd:+.3f}>\n"
        self.last_packet = packet.strip()

        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.write(packet.encode("ascii"))
                mode = f"HW ({self.port_name})"
                connected = True
            except Exception as exc:
                print(f"[WARN] Serial write failed ({exc}); switching to SIM mode.")
                self.disconnect()
                mode = "SIM (Fallback)"
                connected = False
        else:
            mode = "SIM (No HW)"
            connected = False

        return RigTelemetry(
            pan_deg=self.pan_deg,
            tilt_deg=self.tilt_deg,
            connected=connected,
            mode_label=mode,
            last_packet=self.last_packet,
        )