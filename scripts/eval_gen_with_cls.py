import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from datasets.real_video_radar import ACTIONS


DEFAULT_EXTRA_CLASSIFIERS = {
    "efficientnet_b0": os.path.join("outputs", "classifier", "radar_cls_efficientnet_b0", "ckpt", "best.pth"),
    "convnext_tiny": os.path.join("outputs", "classifier", "radar_cls_convnext_tiny", "ckpt", "best.pth"),
    "swin_tiny_patch4_window7_224": os.path.join(
        "outputs", "classifier", "radar_cls_swin_tiny_patch4_window7_224", "ckpt", "best.pth"
    ),
}

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


def build_dataloader(
    root: str, batch_size: int = 32, allow_nested: bool = False, img_size: int = 120
) -> Tuple[DataLoader, List[ImageSample]]:
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = ImageDataset(root=root, transform=transform, allow_nested=allow_nested)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader, dataset.samples


def _parse_extra_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Expected format 'arch=ckpt_path' for --extra_cls_ckpt")
    arch, ckpt = spec.split("=", 1)
    arch = arch.strip()
    ckpt = ckpt.strip()
    if not arch or not ckpt:
        raise argparse.ArgumentTypeError("Classifier arch and checkpoint path must be non-empty")
    return arch, ckpt


def _infer_img_size_from_config(ckpt_path: str, fallback: int) -> int:
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
    return fallback


def load_classifier(ckpt: str, device: torch.device, arch: str = "resnet18") -> torch.nn.Module:
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt}")
    arch = arch.lower()
    if arch in {"resnet", "resnet18"}:
        model = models.resnet18(num_classes=len(ACTIONS))
    else:
        model = timm.create_model(arch, pretrained=False, num_classes=len(ACTIONS))
    state = torch.load(ckpt, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate(
    root: str,
    cls_ckpt: str,
    batch_size: int,
    device: torch.device,
    allow_nested: bool = False,
    debug: bool = False,
    arch: str = "resnet18",
    img_size: int = 120,
) -> Dict:
    loader, samples = build_dataloader(root, batch_size=batch_size, allow_nested=allow_nested, img_size=img_size)
    model = load_classifier(cls_ckpt, device, arch=arch)
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
        "arch": arch,
        "ckpt": cls_ckpt,
        "img_size": img_size,
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
    parser.add_argument("--cls_ckpt", type=str, required=True, help="Primary classifier checkpoint path")
    parser.add_argument(
        "--cls_arch",
        type=str,
        default="resnet18",
        help="Architecture for the primary classifier (default: resnet18)",
    )
    parser.add_argument("--out_json", type=str, required=True, help="Output metrics JSON path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--allow_nested", action="store_true", help="Allow nested subject folders under root")
    parser.add_argument("--debug", action="store_true", help="Output extra debug info (sample paths, confusion)")
    parser.add_argument(
        "--extra_cls_ckpt",
        type=_parse_extra_spec,
        action="append",
        default=[],
        help=(
            "Optional additional classifiers to evaluate, in 'arch=ckpt_path' format. "
            "Example: --extra_cls_ckpt efficientnet_b0=outputs/classifier/radar_cls_efficientnet_b0/ckpt/best.pth"
        ),
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=120,
        help="Fallback image size for classifiers without config metadata (default: 120)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_map = {args.cls_arch: args.cls_ckpt}
    for arch, ckpt in args.extra_cls_ckpt:
        cls_map[arch] = ckpt

    # Automatically append default multi-arch classifiers when their checkpoints exist.
    for arch, ckpt in DEFAULT_EXTRA_CLASSIFIERS.items():
        if arch not in cls_map and os.path.exists(ckpt):
            cls_map[arch] = ckpt

    all_metrics: Dict[str, Dict] = {}
    for arch, ckpt in cls_map.items():
        img_size = _infer_img_size_from_config(ckpt, args.img_size)
        try:
            metrics = evaluate(
                args.root,
                ckpt,
                args.batch_size,
                device,
                allow_nested=args.allow_nested,
                debug=args.debug,
                arch=arch,
                img_size=img_size,
            )
        except FileNotFoundError as exc:
            if arch == args.cls_arch:
                raise
            print(f"[warn] Skip {arch}: {exc}")
            continue
        all_metrics[arch] = metrics

    if args.cls_arch not in all_metrics:
        raise RuntimeError(f"Primary classifier '{args.cls_arch}' did not produce metrics (missing checkpoint?)")

    primary_metrics = all_metrics[args.cls_arch]
    output = dict(primary_metrics)
    output["primary_classifier"] = args.cls_arch
    output["all_classifiers"] = all_metrics

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
