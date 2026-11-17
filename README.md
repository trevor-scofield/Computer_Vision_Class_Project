# Hand Gesture Game Controller

A Python computer vision project that detects hands, classifies gestures, and
sends keyboard controls to a game. It uses Google's MediaPipe Hand Landmarker
for hand detection, handedness, and landmark tracking, plus a ResNet-18 image
classifier trained on cropped hand images.

## Features

- Detect up to two hands and estimate anatomical left/right handedness.
- Display hand boxes, 21 landmarks, predicted gesture, and normalized position.
- Train with HaGRID hand images and personal gesture videos.
- Export per-image hand positions and landmarks to CSV files.
- Send keyboard inputs while a recognized gesture is held.
- Choose or cycle through available cameras.

## Requirements

- Windows 10 or later
- Python 3.10–3.12
- A webcam
- The MediaPipe model file `hand_landmarker.task` in the project root (included
  in this repository)

## Install

Open a terminal in the project folder and create an environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch will use the CPU by default for live prediction. For GPU training,
install a PyTorch build that matches the computer's CUDA setup by following the
[official PyTorch install selector](https://pytorch.org/get-started/locally/).

## Data preparation and training

Training images, videos, and trained model checkpoints are kept locally and are
not required to clone the code. For HaGRID download and license notes, see
[GESTURE_PROJECT.md](GESTURE_PROJECT.md).

### Use personal gesture videos

Place `.mp4` files together in a folder. Supported names include
`LefthandOnefinger.mp4`, `RighthandFist-2.mp4`, `Fist1.mp4`, `peacesign2.mp4`,
and `pointing1.mp4`. Explicit `Lefthand`/`Righthand` names use the named side;
generic names use MediaPipe to estimate the side. Numbered `1` and `2` takes
are split into training and test sets, respectively. Keep clips unmirrored so
MediaPipe's handedness estimate stays consistent.

```powershell
python extract_hand_videos.py --input-dir "C:\path\to\gesture_videos"
python prepare_recorded_hand_data.py --frames Data\recorded_hand_frames --output Data\recorded_hand_split
python prepare_hand_crops.py --data Data\recorded_hand_split --output Data\recorded_hand_crops
```

The crop step writes images under `Data/recorded_hand_crops` and landmark CSVs
alongside them.

### Use HaGRID and combine it with personal videos

After preparing a HaGRID train/test image folder at `Data/hagrid`, crop its five
controller gestures and combine them with personal video crops:

```powershell
python prepare_hand_crops.py --data Data\hagrid --output Data\hagrid_controller_crops --label-with-detected-side --controller-gestures-only
python merge_hand_datasets.py --inputs Data\recorded_hand_crops Data\hagrid_controller_crops --output Data\combined_hand_training
```

The HaGRID labels `fist`, `one`, `peace`, `palm`, and `like` are mapped to the
project's five gestures. Images without a detected hand are skipped; live mode
reports “No hand detected” separately. The preparation and dataset commands
are described in more detail in [GESTURE_PROJECT.md](GESTURE_PROJECT.md).

Train the gesture classifier:

```powershell
python main_gesture.py train --data Data\combined_hand_training --output models\hand_controller_mediapipe.pth --epochs 25
```

The classifier learns from cropped images. Landmark coordinates and estimated
handedness are stored as metadata for analysis and future position-based
controls; they are not currently classifier inputs.

## Run the live controller

```powershell
python main_gesture.py play --model models\hand_controller_mediapipe.pth --camera 0
```

Click or switch to the game window after the controller starts so it receives
the keyboard input. The preview displays the keys currently held. Keys require
three consistent frames by default and release when the gesture is no longer
recognized. To adjust recognition confidence or debounce:

```powershell
python main_gesture.py play --model models\hand_controller_mediapipe.pth --confidence-threshold 0.5 --debounce-frames 3
```

Use `--preview-only` to view detections without sending keys. In the preview,
`C` cycles cameras, `0`–`9` selects a camera index, and `Escape` exits.

## Default controls

| Hand | Gesture | Key |
|---|---|---|
| Left | One finger | W |
| Left | Two fingers | D |
| Left | Fist | S |
| Left | Thumbs up | A |
| Left | Open palm | Space |
| Right | One finger | Enter |
| Right | Two fingers | X |
| Right | Fist | C |
| Right | Thumbs up | Q |
| Right | Open palm | F |
## these were stand in but you can make them what you want.
Mappings are defined in `play_gestures.py` and can be changed there.

## Project files

- `main_gesture.py` — main entry point for training and live control.
- `handTrack.py` — MediaPipe Tasks Hand Landmarker wrapper.
- `play_gestures.py` — live camera prediction and keyboard control.
- `train_gestures.py` — ResNet-18 training.
- `extract_hand_videos.py` — sample labeled frames from videos.
- `prepare_hand_crops.py` — crop hands and export landmark metadata.
- `prepare_hagrid.py` — prepare a balanced HaGRID subset.
- `merge_hand_datasets.py` — merge compatible train/test crop datasets.

## Data and model files

Do not commit private source videos, large datasets, or trained model
checkpoints unless you intend to publish and have checked their privacy and
license terms. HaGRID attribution and license requirements are documented in
`GESTURE_PROJECT.md`.
