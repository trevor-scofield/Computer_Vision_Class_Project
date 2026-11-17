"""Extract labeled, non-mirrored training frames from one video per gesture.

Example:
  python extract_hand_videos.py --input-dir C:\\Users\\TJSCO\\Downloads\\Photos-1-001
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2

from handTrack import HandTracker


def labels_from_filename(path: Path) -> tuple[str | None, str | None]:
    """Return (explicit full label, gesture) for named-side and generic clips."""
    name = re.sub(r"[^a-z]", "", path.stem.lower())
    if "nohand" in name:
        return "no_hand", "no_hand"
    for prefix, side in (("lefthand", "left"), ("righthand", "right")):
        if name.startswith(prefix):
            gesture = name[len(prefix):]
            aliases = {
                "onefinger": "one_finger",
                "twofingers": "two_fingers",
                "openpalm": "open_palm",
                "thumbsup": "thumbs_up",
                "fist": "fist",
            }
            if gesture in aliases:
                normalized = aliases[gesture]
                return f"{side}_{normalized}", normalized
    aliases = {
        "fist": "fist",
        "openpalm": "open_palm",
        "thumbsup": "thumbs_up",
        "peacesign": "two_fingers",
        "pointing": "one_finger",
    }
    # Numbered generic clips get their side from MediaPipe, not their suffix.
    for alias, gesture in aliases.items():
        if name.startswith(alias) and name[len(alias):] in {"", "1", "2"}:
            return None, gesture
    raise ValueError(f"Unrecognized video filename: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder containing the labeled MP4 clips")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Data/recorded_hand_frames"))
    parser.add_argument("--sample-seconds", type=float, default=0.5,
                        help="Seconds between saved frames (default: 0.5)")
    parser.add_argument("--trim-fraction", type=float, default=0.15,
                        help="Skip this fraction at each end of gesture clips")
    args = parser.parse_args()
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")
    if not 0 <= args.trim_fraction < 0.5:
        parser.error("--trim-fraction must be in [0, 0.5)")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Video folder does not exist: {args.input_dir}")

    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4 videos found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "frames.csv"
    total = 0
    existing_sources: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as existing_file:
            existing_sources = {row["source_video"] for row in csv.DictReader(existing_file)}
    manifest_mode = "a" if manifest_path.exists() else "w"
    tracker = HandTracker(max_hands=1, detection_confidence=0.5, running_mode="VIDEO")
    with manifest_path.open(manifest_mode, newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file,
                                fieldnames=("image", "label", "source_video", "time_seconds"))
        if manifest_mode == "w":
            writer.writeheader()
        for video_path in videos:
            if video_path.name in existing_sources:
                print(f"Already extracted; skipping: {video_path.name}")
                continue
            explicit_label, gesture = labels_from_filename(video_path)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Skipping unreadable video: {video_path.name}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                cap.release()
                print(f"Skipping video with invalid metadata: {video_path.name}")
                continue

            stride = max(1, round(fps * args.sample_seconds))
            first = round(frame_count * args.trim_fraction) if explicit_label != "no_hand" else 0
            last = round(frame_count * (1 - args.trim_fraction)) if explicit_label != "no_hand" else frame_count
            saved = 0
            output_label = explicit_label or f"detected_{gesture}"
            frame_index = 0
            while frame_index < last:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index >= first and (frame_index - first) % stride == 0:
                    time_seconds = frame_index / fps
                    label = explicit_label
                    if label is None:
                        hands, _ = tracker.detect(frame, timestamp_ms=round(time_seconds * 1000))
                        if not hands:
                            frame_index += 1
                            continue
                        side = hands[0]["detected_handedness"]
                        if side not in {"left", "right"}:
                            frame_index += 1
                            continue
                        label = f"{side}_{gesture}"
                    output_label = label
                    target_dir = args.output_dir / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{video_path.stem}_{time_seconds:06.2f}s.jpg"
                    image_path = target_dir / filename
                    if cv2.imwrite(str(image_path), frame):
                        writer.writerow({
                            "image": image_path.relative_to(args.output_dir).as_posix(),
                            "label": label,
                            "source_video": video_path.name,
                            "time_seconds": f"{time_seconds:.3f}",
                        })
                        saved += 1
                frame_index += 1
            cap.release()
            total += saved
            print(f"{video_path.name}: {saved} frames -> {output_label}")

    tracker.close()

    print(f"Saved {total} frames and manifest: {manifest_path.resolve()}")
    print("These are frame labels from the video names; visually review the clips "
          "before using them for training.")


if __name__ == "__main__":
    main()
