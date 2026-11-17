"""Make clip-separated train/test folders from extracted personal video frames."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path,
                        default=Path("Data/recorded_hand_frames"))
    parser.add_argument("--output", type=Path,
                        default=Path("Data/recorded_hand_split"))
    args = parser.parse_args()
    manifest = args.frames / "frames.csv"
    if not manifest.is_file():
        raise SystemExit(f"Missing frame manifest: {manifest}")

    split_counts = {"train": {}, "test": {}}
    with manifest.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("The frame manifest contains no images")

    for row in rows:
        source_video = Path(row["source_video"]).stem.lower()
        # Keep complete takes isolated: legacy clips use -2 for the held-out
        # take; the newer generic clips use the 1/2 suffix convention.
        split = "test" if source_video.endswith(("-2", "2")) else "train"
        source_image = args.frames / row["image"]
        if not source_image.is_file():
            raise SystemExit(f"Missing image listed in manifest: {source_image}")
        target_dir = args.output / split / row["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_dir / source_image.name)
        split_counts[split][row["label"]] = split_counts[split].get(row["label"], 0) + 1

    if set(split_counts["train"]) != set(split_counts["test"]):
        raise SystemExit("Training and test clips do not contain the same label folders")
    for split in ("train", "test"):
        details = ", ".join(f"{label}={count}" for label, count
                             in sorted(split_counts[split].items()))
        print(f"{split}: {sum(split_counts[split].values())} images ({details})")
    print(f"Prepared clip-separated data in {args.output.resolve()}")


if __name__ == "__main__":
    main()
