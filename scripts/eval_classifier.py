import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.real_video_radar import ACTIONS
from scripts.train_classifier import RadarActionDataset, build_model, build_transform


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(pretrained=False)
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
) -> Tuple[float, float, Dict[str, float]]:
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
    return total_loss / total_samples, overall_acc, per_class_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a radar classifier with per-class accuracies.")
    parser.add_argument("--root", type=str, default="data", help="Dataset root.")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (train|val|test).")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to classifier checkpoint.")
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device)
    loader = build_loader(
        root=args.root, split=args.split, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    criterion = nn.CrossEntropyLoss()

    loss, acc, per_class = evaluate(model, loader, criterion, device)
    metrics = {"split": args.split, "loss": loss, "acc": acc, "per_class_acc": per_class}

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
