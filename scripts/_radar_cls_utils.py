import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.real_video_radar import ACTIONS


class WeightedFocalLoss(nn.Module):
    """Focal loss with optional per-class alpha weights.

    This is useful for hard-to-learn classes like "box" to focus the optimizer on
    misclassified samples.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            class_alpha = self.alpha.gather(0, targets)
            loss = loss * class_alpha

        return loss.mean()


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


def get_model_default_img_size(model: nn.Module, fallback: int) -> int:
    """Infer expected square input size from a timm model's default_cfg, else fallback."""
    default_cfg = getattr(model, "default_cfg", {}) or {}
    size = default_cfg.get("input_size")
    if isinstance(size, (list, tuple)) and len(size) == 3:
        return int(size[1])
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0])
    return fallback


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
    root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
    sampler_box_factor: float,
) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    train_ds = RadarActionDataset(root=root, split="train", img_size=img_size, augment=True)
    val_ds = RadarActionDataset(root=root, split="val", img_size=img_size, augment=False)

    train_sampler: WeightedRandomSampler | None = None
    if use_weighted_sampler:
        labels = [label for _, label in train_ds.samples]
        class_counts = np.bincount(labels, minlength=len(ACTIONS)).astype(np.float64)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)

        box_idx = ACTIONS.index("box")
        class_weights[box_idx] *= sampler_box_factor

        sample_weights = [class_weights[label] for label in labels]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    test_loader: DataLoader | None = None
    try:
        test_ds = RadarActionDataset(root=root, split="test", img_size=img_size, augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    except RuntimeError:
        test_loader = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not bool(train_sampler),
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


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


def dump_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
