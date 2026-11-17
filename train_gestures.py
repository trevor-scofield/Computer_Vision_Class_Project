"""Fine-tune a compact pretrained image classifier on Data/hagrid."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Data/hagrid"))
    parser.add_argument("--output", type=Path, default=Path("models/hagrid_resnet18.pth"))
    parser.add_argument("--resume-from", type=Path,
                        help="Fine-tune a saved project checkpoint instead of downloading ImageNet weights")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="Limit CPU thread count to avoid oversubscription")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
    train_tf = transforms.Compose([transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=.15, contrast=.15), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    # No horizontal/vertical flips: mirrored or inverted gestures can change
    # label meaning. Keep only appearance augmentation that preserves labels.
    train_ds = datasets.ImageFolder(args.data / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(args.data / "test", transform=eval_tf)
    if train_ds.classes != test_ds.classes:
        raise SystemExit("Train/test class folders differ")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Training on {device}: {len(train_ds)} training images, "
          f"{len(test_ds)} test images, {len(train_ds.classes)} classes, "
          f"{args.epochs} epochs.", flush=True)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("classes") != train_ds.classes:
            raise SystemExit("Checkpoint classes do not match the training folder classes")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Fine-tuning from checkpoint: {args.resume_from}")
    else:
        # ImageNet weights are used when available. Pass --resume-from to train
        # offline from a previously saved project checkpoint.
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"Pretrained weights unavailable ({exc}); initializing from scratch.")
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    progress_interval = max(1, (len(train_dl) + 9) // 10)
    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = seen = 0
        print(f"Starting epoch {epoch + 1}/{args.epochs}", flush=True)
        for batch_index, (images, labels) in enumerate(train_dl, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
            if (batch_index == 1 or batch_index % progress_interval == 0
                    or batch_index == len(train_dl)):
                batch_loss = loss.item()
                progress = 100 * batch_index / len(train_dl)
                print(f"Epoch {epoch + 1}/{args.epochs} | "
                      f"batch {batch_index}/{len(train_dl)} ({progress:.0f}%) | "
                      f"loss={batch_loss:.4f}", flush=True)
        model.eval()
        test_correct = test_seen = 0
        with torch.no_grad():
            for images, labels in test_dl:
                logits = model(images.to(device))
                test_correct += (logits.argmax(1).cpu() == labels).sum().item()
                test_seen += labels.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss/seen:.4f}, "
              f"train_acc={correct/seen:.3f}, test_acc={test_correct/test_seen:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "classes": train_ds.classes}, args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
