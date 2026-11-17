"""Shared Google MediaPipe Tasks Hand Landmarker wrapper."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandTracker:
    """Detect hands and return pixel, normalized, and world landmarks."""

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.5,
                 running_mode: str = "VIDEO", hand_presence_confidence: float = 0.5,
                 tracking_confidence: float = 0.5, model_path: str | Path | None = None):
        root = Path(__file__).resolve().parent
        self.model_path = Path(model_path) if model_path else root / "hand_landmarker.task"
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}. Place hand_landmarker.task in the project root."
            )
        modes = mp.tasks.vision.RunningMode
        mode_names = {"IMAGE": modes.IMAGE, "VIDEO": modes.VIDEO}
        try:
            mode = mode_names[running_mode.upper()]
        except KeyError as exc:
            raise ValueError("running_mode must be IMAGE or VIDEO") from exc
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=hand_presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.running_mode = running_mode.upper()
        self._last_timestamp_ms = -1
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, image, timestamp_ms: int | None = None):
        """Return (hand records, original image); each record has a pixel bbox."""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.running_mode == "IMAGE":
            result = self._landmarker.detect(mp_image)
        else:
            timestamp = timestamp_ms if timestamp_ms is not None else time.monotonic_ns() // 1_000_000
            timestamp = max(int(timestamp), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp
            result = self._landmarker.detect_for_video(mp_image, timestamp)

        hands = []
        for index, points in enumerate(result.hand_landmarks):
            landmarks_norm = [[float(p.x), float(p.y), float(p.z)] for p in points]
            landmarks_px = [[int(round(p.x * width)), int(round(p.y * height)), float(p.z)] for p in points]
            xs = [p[0] for p in landmarks_px]
            ys = [p[1] for p in landmarks_px]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(width - 1, max(xs)), min(height - 1, max(ys))
            box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
            categories = result.handedness[index] if index < len(result.handedness) else []
            category = categories[0] if categories else None
            detected_side = category.category_name if category else "Unknown"
            score = float(category.score) if category else 0.0
            world = result.hand_world_landmarks[index] if index < len(result.hand_world_landmarks) else []
            hands.append({
                "bbox": (x1, y1, box_w, box_h),
                "bbox_norm": [x1 / width, y1 / height, box_w / width, box_h / height],
                "center": [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height],
                "center_norm": [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height],
                "landmarks_norm": landmarks_norm,
                "landmarks_px": landmarks_px,
                "landmarks_hand_norm": self._hand_relative(landmarks_px, x1, y1, box_w, box_h),
                "wrist_norm": landmarks_norm[0][:2] if landmarks_norm else None,
                "detected_handedness": detected_side.lower(),
                "handedness_score": score,
                "world_landmarks_m": [[float(p.x), float(p.y), float(p.z)] for p in world],
            })
        return hands, image

    @staticmethod
    def _hand_relative(points, x, y, width, height):
        return [[(p[0] - x) / width, (p[1] - y) / height, p[2]] for p in points]

    def close(self):
        self._landmarker.close()


def draw_landmarks(image, hand, color=(0, 255, 0)):
    """Draw the 21 MediaPipe landmarks and their standard hand connections."""
    points = hand.get("landmarks_px", [])
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(image, tuple(points[start][:2]), tuple(points[end][:2]), color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point[:2]), 3, color, cv2.FILLED, cv2.LINE_AA)
    return image
