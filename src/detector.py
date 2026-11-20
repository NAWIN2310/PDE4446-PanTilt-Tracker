"""
Object Detection Module for PDE4446 Pan-Tilt Tracker.

Supports:
- Object_A (Tennis Ball) & Object_B (Stress-Ball Football): HSV color segmentation
  filtered by minimum contour area, aspect ratio, and circularity.
- Object_C (Custom Non-Spherical Object): HSV color segmentation with rotated
  minimum-area bounding box (cv2.minAreaRect) so centroid and orientation stay
  consistent even when the object rotates in-plane.
"""

from dataclasses import dataclass
import json
import os
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Container for a single frame's detection output."""

    found: bool
    center_px: Optional[Tuple[int, int]] = None
    norm_error: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    rotated_box: Optional[np.ndarray] = None
    angle_deg: float = 0.0
    mask: Optional[np.ndarray] = None


class ObjectDetector:
    """Real-time color and shape detector for Objects A, B, and C."""

    def __init__(self, config_path: str = os.path.join("data", "objects.json")) -> None:
        self.config_path = config_path
        self.profiles: Dict[str, dict] = self._load_profiles(config_path)
        self.active_key: str = "Object_A"
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.hsv_lower = np.array(self.profiles[self.active_key]["hsv_lower"], dtype=np.uint8)
        self.hsv_upper = np.array(self.profiles[self.active_key]["hsv_upper"], dtype=np.uint8)
        self.min_area = float(self.profiles[self.active_key].get("min_area", 400))
        self.min_circularity = float(self.profiles[self.active_key].get("min_circularity", 0.5))

    @staticmethod
    def _load_profiles(path: str) -> Dict[str, dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Object profile config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_active_object(self, object_key: str) -> None:
        """Switch active target profile (Object_A, Object_B, or Object_C)."""
        if object_key not in self.profiles:
            raise ValueError(f"Unknown object key: {object_key}")
        self.active_key = object_key
        profile = self.profiles[object_key]
        self.hsv_lower = np.array(profile["hsv_lower"], dtype=np.uint8)
        self.hsv_upper = np.array(profile["hsv_upper"], dtype=np.uint8)
        self.min_area = float(profile.get("min_area", 400))
        self.min_circularity = float(profile.get("min_circularity", 0.5))

    def update_thresholds(
        self,
        hsv_lower: Tuple[int, int, int],
        hsv_upper: Tuple[int, int, int],
        min_area: float,
        min_circularity: float,
    ) -> None:
        """Update HSV and shape thresholds dynamically from the live UI."""
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = max(50.0, float(min_area))
        self.min_circularity = max(0.0, min(1.0, float(min_circularity)))

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect the active target in a BGR frame.

        Returns DetectionResult with normalized error in [-1.0, +1.0] where
        (0.0, 0.0) is the exact frame center.
        """
        height, width = frame.shape[:2]
        frame_cx = width / 2.0
        frame_cy = height / 2.0

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if self.hsv_lower[0] <= self.hsv_upper[0]:
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        else:
            low1 = np.array([0, self.hsv_lower[1], self.hsv_lower[2]], dtype=np.uint8)
            high1 = self.hsv_upper
            low2 = self.hsv_lower
            high2 = np.array([179, self.hsv_upper[1], self.hsv_upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, low1, high1), cv2.inRange(hsv, low2, high2))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        profile = self.profiles[self.active_key]
        method = profile.get("method", "hsv_circle")
        ar_min, ar_max = profile.get("aspect_ratio_range", [0.2, 5.0])

        best_contour = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 0.0
            if not (ar_min <= aspect_ratio <= ar_max):
                continue

            score = area * (circularity if method == "hsv_circle" else 1.0)
            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return DetectionResult(found=False, mask=mask)

        moments = cv2.moments(best_contour)
        if moments["m00"] <= 0:
            return DetectionResult(found=False, mask=mask)

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        err_x = float(np.clip((cx - frame_cx) / frame_cx, -1.0, 1.0))
        err_y = float(np.clip((cy - frame_cy) / frame_cy, -1.0, 1.0))

        bbox = cv2.boundingRect(best_contour)
        rotated_box = None
        angle_deg = 0.0

        if method == "hsv_oriented_box":
            rect = cv2.minAreaRect(best_contour)
            angle_deg = float(rect[2])
            rotated_box = np.intp(cv2.boxPoints(rect))

        return DetectionResult(
            found=True,
            center_px=(cx, cy),
            norm_error=(err_x, err_y),
            area=float(cv2.contourArea(best_contour)),
            bbox=bbox,
            rotated_box=rotated_box,
            angle_deg=angle_deg,
            mask=mask,
        )