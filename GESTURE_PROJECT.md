# HaGRID gesture controller starter

The gesture pipeline is separate from the original `Main.py` coursework model.
It classifies the camera frame; `handTrack.py` detects hand landmarks. Start with
classification to prove the full loop, then use the landmarks for cursor or
directional control where position matters.

## Dataset and licensing

HaGRID v2 has 34 labels (33 gestures plus `no_gesture`) and is distributed as
large per-class archives. The official download script retrieves whole archives;
there is no official 5,000-image archive. This means the official method can
require substantial temporary disk space even though `prepare_hagrid.py` keeps
only 5,000 images in the project dataset. Review the repository's CC BY-SA 4.0
variant license and cite HaGRID in any public report or redistribution.

Download the classes from the [HaGRID repository](https://github.com/hukenovs/hagrid/tree/master#downloads)
into a location with enough free space, extract them, and arrange them as
`<extracted-root>/<gesture>/<image>.jpg`. Then create a balanced subset:

```powershell
python prepare_hagrid.py --source D:\datasets\hagrid_dataset --output data\hagrid --count 5000
```

This includes all 34 labels at roughly 147 images per label (the final two get
one extra), sampled deterministically. It makes an 80/20 train/test split.

## Train and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gestures.txt
python train_gestures.py --data data\hagrid --epochs 8
python play_gestures.py
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
