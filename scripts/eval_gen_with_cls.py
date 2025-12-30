import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from datasets.real_video_radar import ACTIONS


def build_dataloader(root: str, batch_size: int = 32) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((120, 120)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.ImageFolder(root=root, transform=transform)
    if not dataset.samples:
        raise RuntimeError(f"No images found under {root}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader


def load_classifier(ckpt: str, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(num_classes=len(ACTIONS))
    state = torch.load(ckpt, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(root: str, cls_ckpt: str, batch_size: int, device: torch.device) -> Dict:
    loader = build_dataloader(root, batch_size=batch_size)
    model = load_classifier(cls_ckpt, device)
    total_samples = 0
    correct_samples = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total_samples += labels.numel()
            correct_samples += (preds == labels).sum().item()
            for a in range(len(ACTIONS)):
                mask = labels == a
                per_class_total[a] += mask.sum().item()
                per_class_correct[a] += ((preds == labels) & mask).sum().item()
    overall_acc = correct_samples / max(1, total_samples)
    per_class_acc = {ACTIONS[i]: per_class_correct[i] / max(1, per_class_total[i]) for i in range(len(ACTIONS))}
    per_class_counts = {ACTIONS[i]: per_class_total[i] for i in range(len(ACTIONS))}
    return {
        "total_samples": total_samples,
        "per_class_samples": per_class_counts,
        "overall": overall_acc,
        "per_class": per_class_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated radar samples with a classifier")
    parser.add_argument("--root", type=str, required=True, help="Directory with generated samples (action subfolders)")
    parser.add_argument("--cls_ckpt", type=str, required=True, help="Classifier checkpoint path")
    parser.add_argument("--out_json", type=str, required=True, help="Output metrics JSON path")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(args.root, args.cls_ckpt, args.batch_size, device)
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
