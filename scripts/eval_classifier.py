import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.real_video_radar import ACTIONS
from scripts.train_classifier import RadarActionDataset, build_model, build_transform


def _infer_arch_from_ckpt_path(ckpt_path: str) -> str | None:
    """Infer classifier arch from checkpoint path (e.g., .../radar_cls_efficientnet_b0/ckpt/best.pth)."""
    parts = os.path.normpath(ckpt_path).split(os.sep)
    for part in parts:
        if part.startswith("radar_cls_"):
            arch = part.replace("radar_cls_", "", 1)
            return arch
    return None


def _infer_img_size_from_config(ckpt_path: str, fallback: int | None) -> int:
    config_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key in ("arch_img_size", "img_size"):
                if key in cfg and isinstance(cfg[key], (int, float)):
                    return int(cfg[key])
        except Exception:  # noqa: BLE001
            pass
    return 120 if fallback is None else fallback


def load_model(ckpt_path: str, device: torch.device, arch: str | None = None) -> torch.nn.Module:
    model_arch = arch or _infer_arch_from_ckpt_path(ckpt_path) or "resnet18"
    if model_arch in {"resnet", "resnet18"}:
        model = build_model(pretrained=False)
    else:
        model = timm.create_model(model_arch, pretrained=False, num_classes=len(ACTIONS))
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_loader(root: str, split: str, img_size: int, batch_size: int, num_workers: int) -> DataLoader:
    ds = RadarActionDataset(root=root, split=split, img_size=img_size, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, Dict[str, float], Dict[str, int], int]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for cls_idx in range(len(ACTIONS)):
            mask = labels == cls_idx
            per_class_total[cls_idx] += mask.sum().item()
            per_class_correct[cls_idx] += ((preds == labels) & mask).sum().item()

    overall_acc = total_correct / max(1, total_samples)
    per_class_acc = {
        ACTIONS[i]: per_class_correct[i] / max(1, per_class_total[i]) for i in range(len(ACTIONS))
    }
    per_class_counts = {ACTIONS[i]: per_class_total[i] for i in range(len(ACTIONS))}
    return total_loss / max(1, total_samples), overall_acc, per_class_acc, per_class_counts, total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a radar classifier with per-class accuracies.")
    parser.add_argument("--root", type=str, default="data", help="Dataset root.")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (train|val|test).")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to classifier checkpoint.")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Classifier architecture (defaults to inferring from checkpoint path, else resnet18).",
    )
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Input image size (defaults to reading config.json near ckpt, else 120).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = _infer_img_size_from_config(args.ckpt, args.img_size)
    model = load_model(args.ckpt, device, arch=args.arch)
    loader = build_loader(
        root=args.root, split=args.split, img_size=img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    criterion = nn.CrossEntropyLoss()

    loss, acc, per_class, per_class_counts, total_samples = evaluate(model, loader, criterion, device)
    metrics = {
        "split": args.split,
        "arch": args.arch or _infer_arch_from_ckpt_path(args.ckpt) or "resnet18",
        "loss": loss,
        "acc": acc,
        "total_samples": total_samples,
        "per_class_samples": per_class_counts,
        "per_class_acc": per_class,
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
