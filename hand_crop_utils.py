"""Shared square hand crop used by training and live prediction."""
from __future__ import annotations

import numpy as np


def square_hand_crop(image: np.ndarray, bbox, margin: float = 0.35) -> np.ndarray | None:
    """Return a padded square crop around a detector's (x, y, width, height) box."""
    x, y, width, height = (int(value) for value in bbox)
    if width <= 0 or height <= 0:
        return None
    side = max(width, height)
    side = max(1, int(round(side * (1 + 2 * margin))))
    center_x = x + width / 2
    center_y = y + height / 2
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    right, bottom = left + side, top + side

    crop = np.zeros((side, side, 3), dtype=image.dtype)
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.shape[1], right), min(image.shape[0], bottom)
    if src_left >= src_right or src_top >= src_bottom:
        return None
    dst_left, dst_top = src_left - left, src_top - top
    crop[dst_top:dst_top + (src_bottom - src_top),
         dst_left:dst_left + (src_right - src_left)] = image[src_top:src_bottom, src_left:src_right]
    return crop


def expanded_box(bbox, image_shape, margin: float = 0.08):
    """Return a lightly expanded, image-clamped box for drawing in the preview."""
    x, y, width, height = (int(value) for value in bbox)
    pad_x, pad_y = round(width * margin), round(height * margin)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_shape[1] - 1, x + width + pad_x)
    bottom = min(image_shape[0] - 1, y + height + pad_y)
    return left, top, right, bottom
