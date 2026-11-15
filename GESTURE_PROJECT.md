# HaGRID gesture controller starter

The gesture project has its own entry point and model pipeline. It classifies
the camera frame; `handTrack.py` detects hand landmarks. Start with
classification to prove the full loop, then use the landmarks for cursor or
directional control where position matters.

## Dataset and licensing

HaGRID v2 has 34 labels (33 gestures plus `no_gesture`) and is distributed as
large per-class archives. The official download script retrieves whole archives;
there is no official 5,000-image archive. This means the official method can
require substantial temporary disk space even though `prepare_hagrid.py` keeps
only 5,000 images in the project dataset. Review the repository's CC BY-SA 4.0
variant license and cite HaGRID in any public report or redistribution.

### Quick start: small HaGRID subset

For an early training run, use the community-hosted HaGRID subset containing up
to 100 images per category (the Hugging Face ZIP is about 402 MB). It should
provide roughly 3,400 images across the 34 labels, enough to validate the
pipeline before investing in more data. It is a third-party subset, not an
official HaGRID release. The host labels it CC BY-SA 4.0; retain HaGRID
attribution and check the original dataset's license terms before redistributing
images or a model trained on them.

```powershell
New-Item -ItemType Directory -Force Data\downloads, Data\hagrid_source | Out-Null
Invoke-WebRequest `
  -Uri "https://huggingface.co/datasets/GestureDetectionConnoisseurs/hagrid_subsets/resolve/main/hagrid-export_100_images.zip" `
  -OutFile "Data\downloads\hagrid-export_100_images.zip"
Expand-Archive -Path "Data\downloads\hagrid-export_100_images.zip" -DestinationPath "Data\hagrid_source" -Force
python main_gesture.py prepare --source Data\hagrid_source --output Data\hagrid --count 3400
python main_gesture.py train --data Data\hagrid --epochs 8
```

The archive page reports a 402 MB ZIP. If the prepared subset reports that a
class has fewer than 100 images, reduce `--count` accordingly or omit that class
from `prepare_hagrid.py` and the controller mapping.

Download the classes from the [HaGRID repository](https://github.com/hukenovs/hagrid/tree/master#downloads)
into a location with enough free space, extract them, and arrange them as
`<extracted-root>/<gesture>/<image>.jpg`. Then create a balanced subset:

```powershell
python prepare_hagrid.py --source D:\datasets\hagrid_dataset --output Data\hagrid --count 5000
```

This includes all 34 labels at roughly 147 images per label (the final two get
one extra), sampled deterministically. It makes an 80/20 train/test split under
`Data/hagrid` by default.

## Main project entry point

Use `main_gesture.py` as the entry point for this project. It dispatches the
three stages below:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gestures.txt
python main_gesture.py prepare --source D:\datasets\hagrid_dataset --output Data\hagrid --count 5000
python main_gesture.py train --data Data\hagrid --epochs 8
python main_gesture.py play --model models\hagrid_resnet18.pth --camera 0 --max-cameras 6
```

While the camera preview is open, press `C` to cycle through camera indexes
`0` to `--max-cameras - 1`, press a number key `0`-`9` to select that camera
index directly, and press `Q` to quit. Camera indexes are provided by OpenCV;
if a camera is unavailable, the preview stays on the current camera.
On Windows, the camera reader tries DirectShow first and then Media Foundation;
the active backend is shown in the preview. If a camera stops returning frames,
quit and restart the controller before selecting another camera.
The live preview also checks for a hand before classifying a gesture and displays
`No hand detected` when the detector does not find one.

Get stage-specific options with `python main_gesture.py train --help` (or
`prepare` / `play`).

Training uses resize and mild color jitter; it does not mirror or invert images,
so directional and inverted gesture labels are preserved.

## Personal video frames

Extract frames from videos named like `LefthandOnefinger.mp4`,
`RighthandFist.mp4`, and `AreaNoHands.mp4` with:

```powershell
python extract_hand_videos.py --input-dir C:\Users\TJSCO\Downloads\Photos-1-001
```

The extractor samples one frame every 0.5 seconds, skips the first and last
15% of gesture clips, and stores labels such as `left_one_finger` and
`right_fist` under `Data/recorded_hand_frames`. `frames.csv` records each
image's source clip and timestamp. These are starting examples, not a reliable
evaluation set: collect several separate takes and lighting conditions before
splitting by clip and training. The current live pipeline still classifies a
whole frame; per-hand cropping and independent two-hand controls are a separate
runtime change.

Prepare and train on two sets of personal clips (original clips for training,
`*-2.mp4` clips held out for testing) with:

```powershell
python prepare_recorded_hand_data.py --frames Data\recorded_hand_frames --output Data\recorded_hand_split
python main_gesture.py train --data Data\recorded_hand_split --output models\hand_side_resnet18.pth --epochs 8
```

The default controller mapping is in `play_gestures.py` (`one`=W, `two_up`=D,
`fist`=S, `palm`=Space, `like`=A). The starter displays the mapped key; it does
not inject keypresses into games. Add `pydirectinput` or another input library
only after checking the target game's input/anti-cheat policy. A temporal
confidence/debounce layer is also recommended before sending live controls.

`handTrack.py` currently depends on cvzone and uses hard-coded coordinate rules
that can misclassify mirrored or rotated hands. HaGRID's image classifier gives
you a trainable baseline first; use MediaPipe/cvzone landmarks separately for
movement and normalize coordinates relative to the detected hand bounding box.
