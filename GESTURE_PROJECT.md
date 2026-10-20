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
python main_gesture.py play --model models\hagrid_resnet18.pth --camera 0
```

Get stage-specific options with `python main_gesture.py train --help` (or
`prepare` / `play`).

The default controller mapping is in `play_gestures.py` (`one`=W, `two_up`=D,
`fist`=S, `palm`=Space, `like`=A). The starter displays the mapped key; it does
not inject keypresses into games. Add `pydirectinput` or another input library
only after checking the target game's input/anti-cheat policy. A temporal
confidence/debounce layer is also recommended before sending live controls.

`handTrack.py` currently depends on cvzone and uses hard-coded coordinate rules
that can misclassify mirrored or rotated hands. HaGRID's image classifier gives
you a trainable baseline first; use MediaPipe/cvzone landmarks separately for
movement and normalize coordinates relative to the detected hand bounding box.
