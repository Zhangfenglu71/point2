import argparse
import os

from engine.train_loop import TrainConfig, run_training


DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rectified flow radar generator")
    parser.add_argument("--exp", type=str, choices=["A_base", "B_cond", "C_full"], required=True)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--clip_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cond_drop", type=float, default=None)
    parser.add_argument("--use_film", type=int, default=0)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--radar_channels", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--enable_cache", type=int, default=1, help="Cache decoded videos in memory")
    parser.add_argument(
        "--preload_videos", type=int, default=0, help="Decode all videos at startup (requires RAM)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_film = bool(args.use_film) or args.exp == "C_full"
    if args.exp == "A_base":
        use_film = False
        cond_drop = 1.0
    elif args.exp == "B_cond":
        use_film = False
        cond_drop = 0.0 if args.cond_drop is None else args.cond_drop
    else:
        cond_drop = 0.25 if args.cond_drop is None else args.cond_drop

    cfg = TrainConfig(
        exp=args.exp,
        root=args.root,
        img_size=args.img_size,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        run_name=args.run_name,
        cond_drop=cond_drop,
        use_film=use_film,
        use_amp=bool(args.use_amp),
        num_workers=args.num_workers,
        radar_channels=args.radar_channels,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        enable_cache=bool(args.enable_cache),
        preload_videos=bool(args.preload_videos),
    )
    os.makedirs("outputs", exist_ok=True)
    run_training(cfg)


if __name__ == "__main__":
    main()
