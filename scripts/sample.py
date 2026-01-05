import argparse
import os

from engine.sample_loop import SampleConfig, run_sampling


DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from rectified flow generator")
    parser.add_argument(
        "--exp",
        type=str,
        choices=["A_base", "B_cond", "C_film", "C_full", "D_full", "E_full", "F_freq"],
        required=True,
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subject", type=str, default="S10")
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--clip_len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--radar_channels", type=int, default=1)
    parser.add_argument("--cfg_w", type=float, default=None)
    parser.add_argument("--cfg_w0", type=float, default=1.0)
    parser.add_argument("--cfg_w1", type=float, default=1.0)
    parser.add_argument("--schedule", type=str, default="const", choices=["const", "linear"])
    parser.add_argument("--num_per_class", type=int, default=64)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints for conditional inputs")
    parser.add_argument("--debug_samples", type=int, default=3, help="Number of samples per class to print debug stats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_run_name = args.run_name or f"sample_{args.exp}"
    exp = args.exp
    # Legacy alias: old C_full maps to E_full checkpoints.
    if exp == "C_full":
        exp = "E_full"
    # F_freq reuses E_full checkpoints and sampling behavior.
    if exp == "F_freq":
        exp = "E_full"
    # Default CFG weight: only guided variants (E_full) use w=3 if user does not override.
    default_cfg_w = 3.0 if exp in {"E_full"} else 1.0
    cfg_w = default_cfg_w if args.cfg_w is None else args.cfg_w
    cfg = SampleConfig(
        exp=exp,
        ckpt_path=args.ckpt,
        root=args.root,
        split=args.split,
        subject=args.subject,
        img_size=args.img_size,
        clip_len=args.clip_len,
        steps=args.steps,
        seed=args.seed,
        run_name=default_run_name,
        radar_channels=args.radar_channels,
        cfg_w=cfg_w,
        cfg_w0=args.cfg_w0,
        cfg_w1=args.cfg_w1,
        schedule=args.schedule,
        num_per_class=args.num_per_class,
        debug=args.debug,
        debug_samples=args.debug_samples,
    )
    os.makedirs("outputs", exist_ok=True)
    run_sampling(cfg)


if __name__ == "__main__":
    main()
