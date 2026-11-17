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
python prepare_hand_crops.py --data Data\hagrid --output Data\hagrid_controller_crops --label-with-detected-side --controller-gestures-only
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
python prepare_hand_crops.py --data Data\hagrid --output Data\hagrid_controller_crops --label-with-detected-side --controller-gestures-only
python main_gesture.py train --data Data\hagrid_controller_crops --output models\hagrid_controller.pth --epochs 25
python main_gesture.py play --model models\hagrid_controller.pth --camera 0 --max-cameras 6
```

While the camera preview is open, press `C` to cycle through camera indexes
`0` to `--max-cameras - 1`, press a number key `0`-`9` to select that camera
index directly, and press `Escape` to quit. Camera indexes are provided by OpenCV;
if a camera is unavailable, the preview stays on the current camera.
On Windows, the camera reader tries DirectShow first and then Media Foundation;
the active backend is shown in the preview. If a camera stops returning frames,
quit and restart the controller before selecting another camera.
The live preview also checks for a hand before classifying a gesture and displays
`No hand detected` when the detector does not find one. Crop HaGRID images with
MediaPipe before training; its `no_gesture` category is excluded because the
live controller already handles hand absence.

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
`right_fist` under `Data/recorded_hand_frames`. Generic names such as
`Fist1.mp4`, `peacesign2.mp4`, and `pointing1.mp4` are also supported; MediaPipe
assigns their left/right label from each detected hand. Numbered `1` clips are
training takes and numbered `2` clips are held out for testing. `frames.csv`
records each image's source clip and timestamp. The preparation script then
uses MediaPipe to crop each detected hand and export its landmark metadata.

The new numbered videos are in the second folder. Extract their frames, split
by take, and create MediaPipe hand crops:

```powershell
python extract_hand_videos.py --input-dir "C:\Users\TJSCO\Downloads\Photos-1-001 (1)"
python prepare_recorded_hand_data.py --frames Data\recorded_hand_frames --output Data\recorded_hand_split_mediapipe
python prepare_hand_crops.py --data Data\recorded_hand_split_mediapipe --output Data\recorded_hand_crops_mediapipe
python prepare_hand_crops.py --data Data\hagrid --output Data\hagrid_controller_crops --label-with-detected-side --controller-gestures-only
python merge_hand_datasets.py --inputs Data\recorded_hand_crops_mediapipe Data\hagrid_controller_crops --output Data\combined_hand_training
python main_gesture.py train --data Data\combined_hand_training --output models\hand_controller_mediapipe.pth --epochs 25
```

HaGRID's `fist`, `one`, `peace`, `palm`, and `like` labels are normalized to
the project's five gestures; MediaPipe supplies their left/right prefixes.
The merge keeps classes with train and test images in both sources. The live
controller resolves generic HaGRID class names with the tracked hand's side.

The crop-preparation step uses the reusable tracker in `handTrack.py`, the same
square crop as live prediction, and writes `landmarks_train.csv` and
`landmarks_test.csv` beside the cropped images. Each row includes normalized
hand-box and center positions, wrist position, and all 21 landmarks in both
frame coordinates and hand-box-relative coordinates, MediaPipe handedness and
its confidence, and 3D hand-relative world landmarks. It skips `no_hand` images
because live mode reports that state directly from hand detection. The live
preview draws each predicted label and normalized center (`x`, `y`) above its
hand box; coordinates run from 0 at the left/top to 1 at the right/bottom. The
preview also shows MediaPipe's anatomical left/right estimate. Keep the camera
frames unmirrored for consistent handedness labels. The included
`hand_landmarker.task` model is used by both data preparation and live mode.

The controller maps left-hand classes to movement (`left_one_finger`=W,
`left_two_fingers`=D, `left_fist`=S, `left_thumbs_up`=A,
`left_open_palm`=Space) and right-hand classes to actions (`right_one_finger`=Enter,
`right_two_fingers`=X, `right_fist`=C, `right_thumbs_up`=Q,
`right_open_palm`=F). It holds the mapped key while the gesture remains stable
and releases it when the hand or gesture is lost. The default debounce requires
three consistent frames. Install dependencies with
`python -m pip install -r requirements-gestures.txt`, focus the game window
after starting the camera controller, and check its `Keys held` status. Use
`--preview-only` to test detection without sending inputs. Escape exits the
preview; C still cycles cameras.

`handTrack.py` wraps Google's MediaPipe Tasks Hand Landmarker in VIDEO mode for
the live camera and IMAGE mode for independent training images. It provides
21 image landmarks, anatomical handedness, and 3D hand-relative world
landmarks. The gesture classifier and tracker remain separate so the gesture
model does not mistake where the hand appears in one recording for what gesture
it is. After switching trackers, regenerate the hand crops and landmark CSVs
before training so they come from the same detector used at runtime.
