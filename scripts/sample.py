import argparse
import os

from engine.sample_loop import SampleConfig, run_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from rectified flow generator")
    parser.add_argument("--exp", type=str, choices=["A_base", "B_cond", "C_full"], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--root", type=str, default="/home/zfl/code/point2/data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subject", type=str, default="S10")
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--clip_len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--radar_channels", type=int, default=1)
    parser.add_argument("--cfg_w", type=float, default=1.0)
    parser.add_argument("--cfg_w0", type=float, default=1.0)
    parser.add_argument("--cfg_w1", type=float, default=1.0)
    parser.add_argument("--schedule", type=str, default="const", choices=["const", "linear"])
    parser.add_argument("--num_per_class", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SampleConfig(
        exp=args.exp,
        ckpt_path=args.ckpt,
        root=args.root,
        split=args.split,
        subject=args.subject,
        img_size=args.img_size,
        clip_len=args.clip_len,
        steps=args.steps,
        seed=args.seed,
        run_name=args.run_name,
        radar_channels=args.radar_channels,
        cfg_w=args.cfg_w,
        cfg_w0=args.cfg_w0,
        cfg_w1=args.cfg_w1,
        schedule=args.schedule,
        num_per_class=args.num_per_class,
    )
    os.makedirs("outputs", exist_ok=True)
    run_sampling(cfg)


if __name__ == "__main__":
    main()
