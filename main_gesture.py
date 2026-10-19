"""Main entry point for the HaGRID hand-gesture game-controller project.

Examples:
  python main_gesture.py prepare --source D:\\datasets\\hagrid_dataset
  python main_gesture.py train --epochs 8
  python main_gesture.py play --camera 0

Options after the command are passed to that stage's script. Run
``python main_gesture.py <command> --help`` for its available options.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = {
    "prepare": "prepare_hagrid.py",
    "train": "train_gestures.py",
    "play": "play_gestures.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare HaGRID data, train the gesture model, or run the webcam controller."
    )
    parser.add_argument("stage", choices=STAGES,
                        help="prepare: make balanced subset; train: fit classifier; play: webcam demo")
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0] not in STAGES:
        parser.error(f"choose one of: {', '.join(STAGES)}")

    stage = argv[0]
    script = Path(__file__).resolve().parent / STAGES[stage]
    command = [sys.executable, str(script), *argv[1:]]
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
