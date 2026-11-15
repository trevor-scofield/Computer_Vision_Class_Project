"""Run webcam gesture classification and map recognized gestures to game keys."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from cvzone.HandTrackingModule import HandDetector
from torch import nn
from torchvision import models, transforms

KEY_MAP = {"one": "w", "two_up": "d", "fist": "s", "palm": "space", "like": "a"}


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
    parser.add_argument("--model", type=Path, default=Path("models/hagrid_resnet18.pth"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-cameras", type=int, default=4,
                        help="Number of camera indexes to cycle through with C (default: 4)")
    args = parser.parse_args()
    if args.max_cameras < 1:
        parser.error("--max-cameras must be at least 1")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    hand_detector = HandDetector(maxHands=2, detectionCon=0.65)
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
            hands, frame = hand_detector.findHands(frame)
            if hands:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    probs = model(prep(rgb).unsqueeze(0))[0].softmax(0)
                score, idx = probs.max(0)
                gesture = classes[idx.item()]
                key = KEY_MAP.get(gesture, "")
                label = f"{gesture} {score.item():.0%}" + (f" -> {key}" if key else "")
                label_color = (40, 230, 80)
            else:
                label = "No hand detected"
                label_color = (0, 180, 255)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .8, label_color, 2)
            controls = (f"Camera {camera_index} ({backend_name}) | C: next camera | "
                        "0-9: select | Q: quit")
            cv2.putText(frame, controls, (20, 72), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 1)
            if notice_frames > 0:
                cv2.putText(frame, notice, (20, 102), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (0, 180, 255), 2)
                notice_frames -= 1
            cv2.imshow("HaGRID game controller (q to quit)", frame)
            pressed = cv2.waitKey(1) & 0xFF
            if pressed in (ord("q"), ord("Q")):
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
