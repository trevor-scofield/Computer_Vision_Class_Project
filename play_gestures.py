"""Run webcam gesture classification and map recognized gestures to game keys."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision import models, transforms

KEY_MAP = {"one": "w", "two_up": "d", "fist": "s", "palm": "space", "like": "a"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/hagrid_resnet18.pth"))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    prep = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                probs = model(prep(rgb).unsqueeze(0))[0].softmax(0)
            score, idx = probs.max(0)
            gesture = classes[idx.item()]
            key = KEY_MAP.get(gesture, "")
            label = f"{gesture} {score.item():.0%}" + (f" -> {key}" if key else "")
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (40, 230, 80), 2)
            cv2.imshow("HaGRID game controller (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
