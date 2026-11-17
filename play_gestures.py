"""Run webcam gesture classification and map recognized gestures to game keys."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision import models, transforms

from hand_crop_utils import expanded_box, square_hand_crop
from handTrack import HandTracker, draw_landmarks

#python main_gesture.py play --model models\hand_controller_mediapipe.pth --camera 0 #Start commnad
KEY_MAP = {
    # Left hand: movement controls.
    ("left", "one_finger"): "W",
    ("left", "one"): "W",  # HaGRID label
    ("left", "two_fingers"): "D",
    ("left", "peace"): "D",  # HaGRID label
    ("left", "fist"): "S",
    ("left", "thumbs_up"): "A",
    ("left", "like"): "A",  # HaGRID label
    ("left", "open_palm"): "SPACE",
    ("left", "palm"): "SPACE",  # HaGRID label
    # Right hand: game actions.
    ("right", "one_finger"): "ENTER",
    ("right", "one"): "ENTER",
    ("right", "two_fingers"): "X",
    ("right", "peace"): "X",
    ("right", "fist"): "B",
    ("right", "thumbs_up"): "Q",
    ("right", "like"): "Q",
    ("right", "open_palm"): "F",
    ("right", "palm"): "F",
}


def mapped_key(gesture: str, detected_side: str) -> str:
    """Resolve personal side-prefixed or generic HaGRID labels to a key."""
    gesture = gesture.lower()
    side = detected_side.lower()
    if gesture.startswith(("left_", "right_")):
        side, gesture = gesture.split("_", 1)
    return KEY_MAP.get((side, gesture), "")


def open_camera(index: int):
    """Try reliable capture backends in order and require a readable first frame."""
    if hasattr(cv2, "CAP_DSHOW"):
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")]
    else:
        backends = [(cv2.CAP_ANY, "automatic")]

    for backend, backend_name in backends:
        candidate = cv2.VideoCapture(index, backend)
        if not candidate.isOpened():
            candidate.release()
            continue
        ok, first_frame = candidate.read()
        if ok:
            return candidate, first_frame, backend_name
        candidate.release()
    return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/hand_side_resnet18.pth"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-cameras", type=int, default=4,
                        help="Number of camera indexes to cycle through with C (default: 4)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Show low-confidence predictions as uncertain (default: 0.5)")
    parser.add_argument("--debounce-frames", type=int, default=3,
                        help="Require this many matching frames before pressing a key")
    parser.add_argument("--preview-only", action="store_true",
                        help="Show key mappings without sending keyboard input")
    args = parser.parse_args()
    if args.max_cameras < 1:
        parser.error("--max-cameras must be at least 1")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be between 0 and 1")
    if args.debounce_frames < 1:
        parser.error("--debounce-frames must be at least 1")
    key_sender = None
    if not args.preview_only:
        try:
            import pydirectinput
        except ImportError as exc:
            raise SystemExit("Keyboard input needs pydirectinput. Install it with: "
                             "python -m pip install pydirectinput") from exc
        pydirectinput.PAUSE = 0
        key_sender = pydirectinput
    key_counts: dict[str, int] = {}
    held_keys: set[str] = set()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    hand_tracker = HandTracker(max_hands=2, detection_confidence=0.65)
    prep = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    cap, pending_frame, backend_name = open_camera(args.camera)
    if cap is None:
        raise SystemExit(f"Could not open camera {args.camera} with DirectShow or Media Foundation")
    camera_index = args.camera
    notice = ""
    notice_frames = 0
    try:
        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if not ok:
                    notice = f"Camera {camera_index} stopped returning frames"
                    notice_frames = 45
                    break
            else:
                frame, pending_frame = pending_frame, None
            source_frame = frame.copy()
            hands, frame = hand_tracker.detect(frame)
            detected_keys: set[str] = set()
            if hands:
                for hand in hands:
                    bbox = hand["bbox"]
                    crop = square_hand_crop(source_frame, bbox)
                    if crop is None:
                        continue
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        probs = model(prep(rgb).unsqueeze(0))[0].softmax(0)
                    score, idx = probs.max(0)
                    gesture = classes[idx.item()]
                    hand_x, hand_y = hand["center_norm"]
                    side = hand["detected_handedness"].title()
                    key = mapped_key(gesture, side)
                    label = (f"{side} {gesture} {score.item():.0%} "
                             f"x:{hand_x:.2f} y:{hand_y:.2f}")
                    if score.item() < args.confidence_threshold:
                        label = f"Uncertain: {label}"
                    elif key:
                        label += f" -> {key}"
                        detected_keys.add(key.lower())

                    x1, y1, x2, y2 = expanded_box(bbox, frame.shape)
                    draw_landmarks(frame, hand)
                    box_color = (40, 230, 80) if score.item() >= args.confidence_threshold else (0, 180, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .6, 2
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
                    text_x = min(x1, frame.shape[1] - text_w - 8)
                    text_y = max(text_h + baseline + 6, y1 - 7)
                    top = text_y - text_h - baseline - 5
                    cv2.rectangle(frame, (text_x, top), (text_x + text_w + 8, text_y + 3),
                                  (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, label, (text_x + 4, text_y - baseline), font, scale,
                                box_color, thickness, cv2.LINE_AA)
            else:
                label = "No hand detected"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            .8, (0, 180, 255), 2)
            for key in set(key_counts) | detected_keys:
                key_counts[key] = key_counts.get(key, 0) + 1 if key in detected_keys else 0
            desired_keys = {key for key, count in key_counts.items()
                            if key in detected_keys and count >= args.debounce_frames}
            if key_sender is not None:
                for key in held_keys - desired_keys:
                    key_sender.keyUp(key)
                for key in desired_keys - held_keys:
                    key_sender.keyDown(key)
            held_keys = desired_keys
            held_label = ", ".join(key.upper() for key in sorted(held_keys)) or "none"
            input_label = ("Preview only" if key_sender is None else f"Keys held: {held_label}")
            cv2.putText(frame, input_label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (40, 230, 80) if held_keys else (220, 220, 220), 2)
            controls = (f"Camera {camera_index} ({backend_name}) | C: next camera | "
                        "0-9: select | Esc: quit")
            cv2.putText(frame, controls, (20, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (255, 255, 255), 1)
            if notice_frames > 0:
                cv2.putText(frame, notice, (20, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (0, 180, 255), 2)
                notice_frames -= 1
            cv2.imshow("Hand gesture game controller (Esc to quit)", frame)
            pressed = cv2.waitKey(1) & 0xFF
            if pressed == 27:  # Escape exits; Q is reserved for the right-hand game action.
                break
            if pressed in (ord("c"), ord("C")):
                camera_indices = [((camera_index + offset) % args.max_cameras)
                                  for offset in range(1, args.max_cameras)]
            elif ord("0") <= pressed <= ord("9"):
                camera_indices = [pressed - ord("0")]
            else:
                continue

            switched = False
            for new_index in camera_indices:
                if new_index == camera_index:
                    continue
                new_cap, first_frame, new_backend = open_camera(new_index)
                if new_cap is None:
                    continue
                cap.release()
                cap = new_cap
                camera_index = new_index
                backend_name = new_backend
                pending_frame = first_frame
                notice = f"Switched to camera {camera_index}"
                notice_frames = 45
                switched = True
                break
            if not switched:
                notice = "No accessible camera at that index"
                notice_frames = 45
    finally:
        if key_sender is not None:
            for key in held_keys:
                key_sender.keyUp(key)
        cap.release()
        hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
