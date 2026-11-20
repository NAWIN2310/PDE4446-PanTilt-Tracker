# PDE4446 Motion Control — Real-Time 2-DOF Pan-Tilt Visual Tracker

A modular computer vision and closed-loop motion control pipeline built in Python and OpenCV for tracking multiple colored and non-spherical targets (`Object_A`, `Object_B`, `Object_C`) across a 2-axis pan-tilt mechanism.

## Architecture
- **`src/detector.py`**: HSV segmentation with morphological filtering, shape circularity checks (`Object_A`/`Object_B`), and oriented bounding boxes via `cv2.minAreaRect` (`Object_C`). Outputs normalized center error in `[-1.0, +1.0]`.
- **`src/kalman_tracker.py`**: 2D constant-velocity Kalman filter (`[px, py, vx, vy]`) that reduces centroid jitter and coasts smoothly through brief occlusions.
- **`src/controller.py`**: Dual-axis PID controller with deadband, anti-windup clamping, low-pass derivative filtering, and normalized outputs `[-1.0, +1.0]`.
- **`src/serial_comm.py`**: USB-Serial protocol (`<PAN:+0.000,TILT:+0.000>\n` at 115200 baud) with automatic fallback to an internal kinematic simulation rig.
- **`src/ui.py` & `src/main.py`**: Real-time HUD overlay, Picture-in-Picture binary mask, hotkeys (`1/2/3`, `L`, `M`, `C`, `Q`), and CSV telemetry logging into `logs/`.
- **`tools/hsv_tuner.py`**: Live HSV and shape calibration utility that writes directly to `data/objects.json`.
- **`tools/mock_rig.py` & `tools/video_replay.py`**: Offline closed-loop simulator and video benchmark plotter.

## Quick Start
```bash
conda activate pantilt
python -m src.main