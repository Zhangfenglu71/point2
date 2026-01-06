import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import timm
import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.real_video_radar import ACTIONS
from scripts._radar_cls_utils import (
    WeightedFocalLoss,
    build_dataloaders,
    dump_json,
    evaluate,
    get_model_default_img_size,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple radar action classifiers (EfficientNet/ConvNeXt/Swin) sequentially."
    )
    parser.add_argument("--root", type=str, default="data", help="Dataset root (train/val[/test] splits).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Lower bound for LR scheduler.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", type=int, default=1, help="Use ImageNet weights for the backbone.")
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=2,
        help="Patience (epochs) for ReduceLROnPlateau when monitoring val_loss.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Stop training if val_loss does not improve for N epochs.",
    )
    parser.add_argument(
        "--class_weight_box",
        type=float,
        default=1.1,
        help="Optional up-weighting factor for the 'box' class in the loss (1.0 to disable).",
    )
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        help="Use focal loss (with optional alpha for the 'box' class) instead of plain cross entropy.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss.",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Use a weighted random sampler to oversample the 'box' class during training.",
    )
    parser.add_argument(
        "--sampler_box_factor",
        type=float,
        default=1.5,
        help="Multiplicative factor applied to the 'box' class in the sampler weights.",
    )
    parser.add_argument(
        "--archs",
        type=str,
        default="efficientnet_b0,convnext_tiny,swin_tiny_patch4_window7_224",
        help="Comma separated list of timm architectures to train sequentially.",
    )
    parser.add_argument(
        "--hf_hub_download_timeout",
        type=int,
        default=60,
        help="Timeout (seconds) for HuggingFace hub model downloads. Ignored if --pretrained 0.",
    )
    return parser.parse_args()


def configure_hf_env(args: argparse.Namespace) -> None:
    if args.pretrained:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hf_hub_download_timeout))
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.hf_hub_download_timeout))
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def build_model(arch: str, args: argparse.Namespace) -> nn.Module:
    try:
        model = timm.create_model(arch, pretrained=bool(args.pretrained), num_classes=len(ACTIONS))
        return model
    except Exception as exc:  # noqa: BLE001
        if args.pretrained:
            print(
                f"[{arch}] pretrained weights download/load failed ({exc}); "
                "falling back to random init (pretrained=False)."
            )
            model = timm.create_model(arch, pretrained=False, num_classes=len(ACTIONS))
            return model
        raise


def train_single_arch(args: argparse.Namespace, arch: str) -> Dict[str, float | int | str]:
    run_name = f"radar_cls_{arch}"
    out_dir = os.path.join("outputs", "classifier", run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, args)
    arch_img_size = get_model_default_img_size(model, fallback=args.img_size)
    config = vars(args) | {"arch": arch, "run_name": run_name, "arch_img_size": arch_img_size}
    dump_json(os.path.join(out_dir, "config.json"), config)

    train_loader, val_loader, test_loader = build_dataloaders(
        root=args.root,
        img_size=arch_img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
        sampler_box_factor=args.sampler_box_factor,
    )

    model = model.to(device)

    # Loss: choose between cross entropy and focal. Both support box reweighting.
    class_weights = torch.ones(len(ACTIONS), device=device)
    if args.class_weight_box != 1.0:
        class_weights[ACTIONS.index("box")] = args.class_weight_box

    if args.focal_loss:
        criterion = WeightedFocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights if args.class_weight_box != 1.0 else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"[{arch}][epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_epochs = 0
            ckpt_path = os.path.join(out_dir, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": {"val_loss": val_loss}}, ckpt_path)
            print(f"[{arch}][checkpoint] Saved {ckpt_path}")
        else:
            no_improve_epochs += 1

        if args.early_stop_patience and no_improve_epochs >= args.early_stop_patience:
            print(
                f"[{arch}][early stop] No val improvement for {no_improve_epochs} epochs, "
                f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f}"
            )
            break

    # Reload best weights for final evaluation and saving
    if best_epoch != -1:
        ckpt = torch.load(os.path.join(out_dir, "ckpt", "best.pth"), map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    metrics: Dict[str, float | int] = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "arch_img_size": arch_img_size,
    }

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        metrics.update({"test_loss": test_loss, "test_acc": test_acc})
        print(f"[{arch}][test] loss={test_loss:.4f}, acc={test_acc:.4f}")

    dump_json(os.path.join(out_dir, "metrics.json"), metrics)
    print(f"[{arch}] best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f}")

    ckpt_path = os.path.join(out_dir, "ckpt", "best.pth") if best_epoch != -1 else ""
    return {
        "arch": arch,
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "ckpt": ckpt_path,
        "metrics_json": os.path.join(out_dir, "metrics.json"),
        "arch_img_size": arch_img_size,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_hf_env(args)

    archs: List[str] = [a.strip() for a in args.archs.split(",") if a.strip()]
    if not archs:
        raise ValueError("No architectures provided via --archs")

    summary: Dict[str, Dict[str, float | int | str]] = {}
    for arch in archs:
        results = train_single_arch(args, arch)
        summary[arch] = results

    summary_path = os.path.join("outputs", "classifier", "radar_cls_multi_summary.json")
    dump_json(summary_path, summary)
    print(f"[summary] Saved multi-arch results to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
