"""Create a deterministic, balanced HaGRID image subset from extracted class ZIPs.

HaGRID's official downloads are complete per-class archives. This script keeps
only the requested number of images in the project working set after extraction.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

CLASSES = [
    "call", "dislike", "fist", "four", "grabbing", "grip", "hand_heart",
    "hand_heart2", "holy", "like", "little_finger", "middle_finger", "mute",
    "no_gesture", "ok", "one", "palm", "peace", "peace_inverted", "point",
    "rock", "stop", "stop_inverted", "take_picture", "three", "three2",
    "three3", "three_gun", "thumb_index", "thumb_index2", "timeout",
    "two_up", "two_up_inverted", "xsign",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted HaGRID folder containing one folder per gesture")
    parser.add_argument("--output", type=Path, default=Path("Data/hagrid"))
    parser.add_argument("--count", type=int, default=5000, help="Total images (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < len(CLASSES):
        parser.error(f"--count must be at least {len(CLASSES)} to include every class")

    rng = random.Random(args.seed)
    base, remainder = divmod(args.count, len(CLASSES))
    # Randomize which classes receive the extra remainder examples.
    class_order = CLASSES.copy()
    rng.shuffle(class_order)
    quotas = {name: base + (i < remainder) for i, name in enumerate(class_order)}
    args.output.mkdir(parents=True, exist_ok=True)
    copied = 0
    for label in CLASSES:
        source_dir = args.source / label
        # The upstream download helper has historically spelled this archive
        # folder "hands_heart2" while annotations and labels use "hand_heart2".
        if label == "hand_heart2" and not source_dir.is_dir():
            source_dir = args.source / "hands_heart2"
        if not source_dir.is_dir():
            raise SystemExit(f"Missing class folder: {source_dir}")
        files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(files)
        chosen = files[:quotas[label]]
        if len(chosen) < quotas[label]:
            raise SystemExit(f"{label}: found {len(chosen)} images, need {quotas[label]}")
        for split, batch in (("train", chosen[:int(len(chosen) * .8)]),
                             ("test", chosen[int(len(chosen) * .8):])):
            target = args.output / split / label
            target.mkdir(parents=True, exist_ok=True)
            for src in batch:
                shutil.copy2(src, target / src.name)
                copied += 1
        print(f"{label}: {len(chosen)} images")
    print(f"Prepared {copied} images in {args.output.resolve()} (seed {args.seed}).")


if __name__ == "__main__":
    main()
