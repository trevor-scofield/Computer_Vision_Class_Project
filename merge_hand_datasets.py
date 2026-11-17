"""Merge cropped hand datasets that share gesture class names."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def images_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Hand-crop datasets, each containing train/ and test/")
    parser.add_argument("--output", type=Path, default=Path("Data/combined_hand_training"))
    args = parser.parse_args()
    for dataset in args.inputs:
        if not all((dataset / split).is_dir() for split in ("train", "test")):
            raise SystemExit(f"Expected train/ and test/ under {dataset}")

    class_names = set.intersection(*[
        {folder.name for folder in (dataset / "train").iterdir() if folder.is_dir()}
        & {folder.name for folder in (dataset / "test").iterdir() if folder.is_dir()}
        for dataset in args.inputs
    ])
    class_names = {name for name in class_names
                   if any(images_in(dataset / "train" / name) for dataset in args.inputs)
                   and any(images_in(dataset / "test" / name) for dataset in args.inputs)}
    if not class_names:
        raise SystemExit("The input datasets have no shared gesture classes with train and test images")

    for split in ("train", "test"):
        for class_name in sorted(class_names):
            target_dir = args.output / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_index, dataset in enumerate(args.inputs):
                source_dir = dataset / split / class_name
                if not source_dir.is_dir():
                    continue
                prefix = f"source{source_index + 1}_"
                for image in images_in(source_dir):
                    shutil.copy2(image, target_dir / f"{prefix}{image.name}")
    print(f"Merged {len(class_names)} shared classes into {args.output.resolve()}")
    print(", ".join(sorted(class_names)))


if __name__ == "__main__":
    main()
