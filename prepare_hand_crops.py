"""Prepare hand-only train/test images using the same detector as live mode."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from hand_crop_utils import square_hand_crop
from handTrack import HandTracker

HAGRID_CONTROL_LABELS = {
    "fist": "fist",
    "one": "one_finger",
    "peace": "two_fingers",
    "palm": "open_palm",
    "like": "thumbs_up",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Data/recorded_hand_split"))
    parser.add_argument("--output", type=Path, default=Path("Data/recorded_hand_crops"))
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--label-with-detected-side", action="store_true",
                        help="Prefix generic dataset labels with MediaPipe left/right")
    parser.add_argument("--controller-gestures-only", action="store_true",
                        help="For HaGRID, keep only fist/one/peace/palm/like controls")
    args = parser.parse_args()
    tracker = HandTracker(max_hands=1, detection_confidence=args.detection_confidence,
                          running_mode="IMAGE")
    counts = {"train": {}, "test": {}}

    for split in ("train", "test"):
        split_dir = args.data / split
        if not split_dir.is_dir():
            raise SystemExit(f"Missing split folder: {split_dir}")
        metadata_path = args.output / f"landmarks_{split}.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
            fields = ["image", "label", "source_image", "frame_width", "frame_height",
                      "center_x_norm", "center_y_norm", "bbox_x_norm", "bbox_y_norm",
                      "bbox_width_norm", "bbox_height_norm", "wrist_x_norm", "wrist_y_norm",
                      "landmarks_norm_json", "landmarks_hand_norm_json",
                      "landmarks_px_json", "world_landmarks_m_json",
                      "detected_handedness", "handedness_score"]
            writer = csv.DictWriter(metadata_file, fieldnames=fields)
            writer.writeheader()
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                # Live mode already reports the absence of a detected hand;
                # HaGRID's no_gesture class is not an isolated hand pose.
                if class_dir.name in {"no_hand", "no_gesture"}:
                    continue
                if args.controller_gestures_only and class_dir.name not in HAGRID_CONTROL_LABELS:
                    continue
                saved = 0
                for source in sorted(class_dir.glob("*.jpg")):
                    image = cv2.imread(str(source))
                    if image is None:
                        continue
                    hands, _ = tracker.detect(image)
                    if not hands:
                        continue
                    hand = hands[0]
                    label = class_dir.name
                    if args.label_with_detected_side:
                        side = hand["detected_handedness"]
                        if side not in {"left", "right"}:
                            continue
                        label = f"{side}_{HAGRID_CONTROL_LABELS.get(label, label)}"
                    target_dir = args.output / split / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    crop = square_hand_crop(image, hand["bbox"])
                    if crop is None:
                        continue
                    target_image = target_dir / source.name
                    if not cv2.imwrite(str(target_image), crop):
                        continue
                    height, width = image.shape[:2]
                    wrist = hand["wrist_norm"] or [None, None]
                    writer.writerow({
                        "image": target_image.relative_to(args.output).as_posix(),
                        "label": label,
                        "source_image": source.relative_to(args.data).as_posix(),
                        "frame_width": width,
                        "frame_height": height,
                        "center_x_norm": hand["center_norm"][0],
                        "center_y_norm": hand["center_norm"][1],
                        "bbox_x_norm": hand["bbox_norm"][0],
                        "bbox_y_norm": hand["bbox_norm"][1],
                        "bbox_width_norm": hand["bbox_norm"][2],
                        "bbox_height_norm": hand["bbox_norm"][3],
                        "wrist_x_norm": wrist[0],
                        "wrist_y_norm": wrist[1],
                        "landmarks_norm_json": json.dumps(hand["landmarks_norm"]),
                        "landmarks_hand_norm_json": json.dumps(hand["landmarks_hand_norm"]),
                        "landmarks_px_json": json.dumps(hand["landmarks_px"]),
                        "world_landmarks_m_json": json.dumps(hand["world_landmarks_m"]),
                        "detected_handedness": hand["detected_handedness"],
                        "handedness_score": hand["handedness_score"],
                    })
                    saved += 1
                counts[split][class_dir.name] = saved
                print(f"{split}/{class_dir.name}: {saved} hand crops + landmarks", flush=True)
        print(f"Saved normalized position and landmark metadata: {metadata_path}")

    if set(counts["train"]) != set(counts["test"]):
        raise SystemExit("Training and test folders do not have matching gesture classes")
    if any(count == 0 for split_counts in counts.values() for count in split_counts.values()):
        raise SystemExit("At least one class has no detected hand crops; check the source frames")
    print(f"Prepared hand crops in {args.output.resolve()}")


if __name__ == "__main__":
    main()
