import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from datasets.real_video_radar import ACTIONS


ImageSample = Tuple[str, int]


def _collect_samples(root: str, allow_nested: bool) -> List[ImageSample]:
    samples: List[ImageSample] = []
    if not allow_nested:
        dataset = datasets.ImageFolder(root=root)
        samples.extend(dataset.samples)
        return samples

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for dirpath, _, filenames in os.walk(root):
        label_idx = None
        parts = dirpath.split(os.sep)
        for part in reversed(parts):
            if part in ACTIONS:
                label_idx = ACTIONS.index(part)
                break
        if label_idx is None:
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                samples.append((os.path.join(dirpath, fname), label_idx))
    return samples


class ImageDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, root: str, transform: transforms.Compose, allow_nested: bool = False) -> None:
        super().__init__()
        self.samples = _collect_samples(root, allow_nested)
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = datasets.folder.default_loader(path)
        return self.transform(img), label


def build_dataloader(root: str, batch_size: int = 32, allow_nested: bool = False) -> Tuple[DataLoader, List[ImageSample]]:
    transform = transforms.Compose(
        [
            transforms.Resize((120, 120)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = ImageDataset(root=root, transform=transform, allow_nested=allow_nested)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader, dataset.samples


def load_classifier(ckpt: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt}")
    model = models.resnet18(num_classes=len(ACTIONS))
    state = torch.load(ckpt, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(
    root: str, cls_ckpt: str, batch_size: int, device: torch.device, allow_nested: bool = False, debug: bool = False
) -> Dict:
    loader, samples = build_dataloader(root, batch_size=batch_size, allow_nested=allow_nested)
    model = load_classifier(cls_ckpt, device)
    total_samples = 0
    correct_samples = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    pred_hist = defaultdict(int)
    confusion = torch.zeros((len(ACTIONS), len(ACTIONS)), dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for p in preds.tolist():
                pred_hist[p] += 1
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t, p] += 1
            total_samples += labels.numel()
            correct_samples += (preds == labels).sum().item()
            for a in range(len(ACTIONS)):
                mask = labels == a
                per_class_total[a] += mask.sum().item()
                per_class_correct[a] += ((preds == labels) & mask).sum().item()
    overall_acc = correct_samples / max(1, total_samples)
    per_class_acc = {ACTIONS[i]: per_class_correct[i] / max(1, per_class_total[i]) for i in range(len(ACTIONS))}
    per_class_counts = {ACTIONS[i]: per_class_total[i] for i in range(len(ACTIONS))}
    metrics = {
        "total_samples": total_samples,
        "per_class_samples": per_class_counts,
        "overall": overall_acc,
        "per_class": per_class_acc,
        "pred_hist": {ACTIONS[i]: pred_hist[i] for i in range(len(ACTIONS))},
        "confusion": confusion.tolist(),
    }
    if debug:
        first_samples = []
        for path, label in samples[:10]:
            first_samples.append({"path": path, "label_id": label, "label_name": ACTIONS[label]})
        metrics["debug_first_samples"] = first_samples
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated radar samples with a classifier")
    parser.add_argument("--root", type=str, required=True, help="Directory with generated samples (action subfolders)")
    parser.add_argument("--cls_ckpt", type=str, required=True, help="Classifier checkpoint path")
    parser.add_argument("--out_json", type=str, required=True, help="Output metrics JSON path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--allow_nested", action="store_true", help="Allow nested subject folders under root")
    parser.add_argument("--debug", action="store_true", help="Output extra debug info (sample paths, confusion)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(args.root, args.cls_ckpt, args.batch_size, device, allow_nested=args.allow_nested, debug=args.debug)
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
