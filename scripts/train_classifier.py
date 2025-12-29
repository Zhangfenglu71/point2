import argparse
import json
import os
import random
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from datasets.real_video_radar import ACTIONS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(img_size: int, augment: bool) -> transforms.Compose:
    ops = [transforms.Grayscale(num_output_channels=3)]
    if augment:
        ops.extend(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        ops.append(transforms.Resize((img_size, img_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


class RadarActionDataset(Dataset):
    """Radar-only action classification dataset using the project layout."""

    def __init__(self, root: str, split: str, img_size: int, augment: bool = False) -> None:
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []

        base = os.path.join(root, split, "radar")
        for subject in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            subject_dir = os.path.join(base, subject)
            if not os.path.isdir(subject_dir):
                continue
            for action in ACTIONS:
                action_dir = os.path.join(subject_dir, action)
                if not os.path.isdir(action_dir):
                    continue
                for fname in sorted(os.listdir(action_dir)):
                    fpath = os.path.join(action_dir, fname)
                    if os.path.isfile(fpath):
                        self.samples.append((fpath, ACTIONS.index(action)))

        if not self.samples:
            raise RuntimeError(
                f"No radar images found at {base}. Expected layout: {base}/Sxx/<action>/*.jpg"
            )

        self.transform = build_transform(img_size=img_size, augment=augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)


def build_dataloaders(
    root: str, img_size: int, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    train_ds = RadarActionDataset(root=root, split="train", img_size=img_size, augment=True)
    val_ds = RadarActionDataset(root=root, split="val", img_size=img_size, augment=False)

    test_loader: DataLoader | None = None
    try:
        test_ds = RadarActionDataset(root=root, split="test", img_size=img_size, augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    except RuntimeError:
        test_loader = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def build_model(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(ACTIONS))
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / max(1, total_samples)


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: dict,
    out_dir: str,
    filename: str = "best.pth",
) -> None:
    ckpt_path = os.path.join(out_dir, "ckpt", filename)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics}, ckpt_path)
    print(f"[checkpoint] Saved {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a radar action classifier for sample evaluation.")
    parser.add_argument("--root", type=str, default="data", help="Dataset root (train/val[/test] splits).")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for outputs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", type=int, default=1, help="Use ImageNet weights for the backbone.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"radar_cls_resnet18_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("outputs", "classifier", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(
        root=args.root, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = build_model(pretrained=bool(args.pretrained)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model=model,
                epoch=epoch,
                metrics={"val_acc": val_acc, "val_loss": val_loss},
                out_dir=out_dir,
                filename="best.pth",
            )

    # Reload best weights for final evaluation and saving
    if best_epoch != -1:
        ckpt = torch.load(os.path.join(out_dir, "ckpt", "best.pth"), map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    metrics = {"val_loss": val_loss, "val_acc": val_acc, "best_epoch": best_epoch}

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        metrics.update({"test_loss": test_loss, "test_acc": test_acc})
        print(f"[test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
