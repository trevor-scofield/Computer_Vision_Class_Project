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
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.15, contrast=.15), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    train_ds = datasets.ImageFolder(args.data / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(args.data / "test", transform=eval_tf)
    if train_ds.classes != test_ds.classes:
        raise SystemExit("Train/test class folders differ")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # No pretrained weights download is required; training starts from ImageNet weights
    # only if torchvision has them cached. Set weights=None for fully offline runs.
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
    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = seen = 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        model.eval()
        test_correct = test_seen = 0
        with torch.no_grad():
            for images, labels in test_dl:
                logits = model(images.to(device))
                test_correct += (logits.argmax(1).cpu() == labels).sum().item()
                test_seen += labels.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss/seen:.4f}, "
              f"train_acc={correct/seen:.3f}, test_acc={test_correct/test_seen:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "classes": train_ds.classes}, args.output)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
