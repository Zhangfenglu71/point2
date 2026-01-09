import argparse
import os

from engine.train_loop import TrainConfig, run_training


DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rectified flow radar generator")
    parser.add_argument(
        "--exp",
        type=str,
        choices=[
            "A_base",
            "B_cond",
            "C_film",
            "C_full",
            "D_full",
            "E_full",
            "F_freq",
            "G_grad",
            "H_taware",
            "K_color",
            "GAN_vid2vid",
            "DIFF_3DUNet",
            "DIFF_STAttn",
            "DIFF_AttnCtrl",
            "DIFF_SegAttn",
        ],
        required=True,
    )
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=120)
    parser.add_argument("--clip_len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["const", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cond_drop", type=float, default=None)
    parser.add_argument("--use_film", type=int, default=0)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--radar_channels", type=int, default=3)
    parser.add_argument("--use_vae", type=int, default=1)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--vae_beta", type=float, default=0.1)
    parser.add_argument("--vae_num_downsamples", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--enable_cache", type=int, default=1, help="Cache decoded videos in memory")
    parser.add_argument(
        "--cache_in_workers",
        type=int,
        default=0,
        help="Allow caching even when using DataLoader workers (may explode RAM).",
    )
    parser.add_argument(
        "--preload_videos", type=int, default=0, help="Decode all videos at startup (requires RAM)"
    )
    parser.add_argument("--freq_lambda", type=float, default=0.0, help="Weight for frequency-band loss")
    parser.add_argument(
        "--freq_band_split1",
        type=float,
        default=1.0 / 3.0,
        help="First normalized split between low/mid frequency bands (0-1)",
    )
    parser.add_argument(
        "--freq_band_split2",
        type=float,
        default=2.0 / 3.0,
        help="Second normalized split between mid/high frequency bands (0-1)",
    )
    parser.add_argument("--debug_freq", type=int, default=0, help="Debug frequency stats interval (steps)")
    parser.add_argument("--grad_lambda", type=float, default=0.0, help="Weight for spectral gradient loss")
    parser.add_argument("--grad_mode", type=str, default="finite_diff", help="Gradient loss mode")
    parser.add_argument(
        "--grad_on", type=str, default="cond_only", choices=["cond_only", "all"], help="Apply grad loss on which samples"
    )
    parser.add_argument("--debug_grad", type=int, default=0, help="Debug grad stats interval (steps)")
    parser.add_argument("--taware", type=int, default=0, help="Enable t-aware weighting for structure losses")
    parser.add_argument("--t_low", type=float, default=0.3, help="Lower t threshold for t-aware mixing")
    parser.add_argument("--t_high", type=float, default=0.7, help="Upper t threshold for t-aware mixing")
    parser.add_argument("--t_mix_power", type=float, default=1.0, help="Exponent for t-aware mixing curve")
    parser.add_argument("--use_action_head", type=int, default=1, help="Enable action classification head on latent")
    parser.add_argument("--action_head_dropout", type=float, default=0.1, help="Dropout rate for action head")
    parser.add_argument("--action_head_dim", type=int, default=256, help="Hidden dim for action head")
    parser.add_argument("--action_adv", type=int, default=0, help="Enable AC-GAN discriminator")
    parser.add_argument("--adv_lambda", type=float, default=0.0, help="Weight for generator adversarial loss")
    parser.add_argument("--perc_lambda", type=float, default=0.0, help="Weight for perceptual loss")
    parser.add_argument("--ssim_lambda", type=float, default=0.0, help="Weight for SSIM reconstruction term")
    parser.add_argument("--recon_lambda", type=float, default=1.0, help="Weight for L1 reconstruction term")
    parser.add_argument("--band_l1_lambda", type=float, default=0.0, help="Weight for frequency band L1 loss")
    parser.add_argument("--temporal_lambda", type=float, default=0.0, help="Weight for temporal smoothness loss")
    parser.add_argument("--infonce_lambda", type=float, default=0.0, help="Weight for InfoNCE contrastive loss")
    parser.add_argument(
        "--contrast_start_epoch", type=int, default=0, help="Epoch to start applying contrastive loss (inclusive)"
    )
    parser.add_argument("--adv_start_epoch", type=int, default=0, help="Epoch to start applying adversarial loss")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for generator weights")
    parser.add_argument("--detect_nan", type=int, default=0, help="Enable NaN/Inf checks during training")
    parser.add_argument(
        "--video_encoder_type",
        type=str,
        default="temporal_unet",
        choices=["timesformer", "video_swin", "vit3d", "cnn", "temporal_unet"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp = args.exp
    if exp in {"GAN_vid2vid", "DIFF_3DUNet", "DIFF_STAttn", "DIFF_AttnCtrl", "DIFF_SegAttn"}:
        run_name = f"train_{exp}"
        if exp == "GAN_vid2vid":
            from engine.video_baseline_gan import GanTrainConfig, run_gan_training

            cfg = GanTrainConfig(
                exp=exp,
                root=args.root,
                img_size=args.img_size,
                clip_len=args.clip_len,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                run_name=run_name,
                use_amp=bool(args.use_amp),
                num_workers=args.num_workers,
                radar_channels=args.radar_channels,
                video_encoder_type=args.video_encoder_type,
                adv_lambda=args.adv_lambda if args.adv_lambda > 0 else 1.0,
                recon_lambda=args.recon_lambda,
            )
            os.makedirs("outputs", exist_ok=True)
            run_gan_training(cfg)
            return
        from engine.video_baseline_diffusion import DiffusionTrainConfig, run_diffusion_training

        cfg = DiffusionTrainConfig(
            exp=exp,
            root=args.root,
            img_size=args.img_size,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            run_name=run_name,
            use_amp=bool(args.use_amp),
            num_workers=args.num_workers,
            radar_channels=args.radar_channels,
            video_encoder_type=args.video_encoder_type,
        )
        os.makedirs("outputs", exist_ok=True)
        run_diffusion_training(cfg)
        return
    # Legacy alias: old C_full now maps to E_full (FiLM + CrossAttn + CFG).
    if exp == "C_full":
        exp = "E_full"
    if exp == "F_freq":
        # F_freq builds on E_full
        exp = "F_freq"

    # Experiment presets
    use_film = bool(args.use_film) or exp in {"C_film", "D_full", "E_full", "F_freq", "G_grad", "H_taware", "K_color"}
    use_cross_attn = exp in {"D_full", "E_full", "F_freq", "G_grad", "H_taware", "K_color"}
    if exp == "A_base":
        use_film = False
        use_cross_attn = False
        cond_drop = 1.0
    elif exp == "B_cond":
        use_film = False
        use_cross_attn = False
        cond_drop = 0.0 if args.cond_drop is None else args.cond_drop
    elif exp == "C_film":
        use_cross_attn = False
        cond_drop = 0.0
    elif exp == "D_full":
        # New D: FiLM + CrossAttn, no CFG training/dropout by default.
        cond_drop = 0.0 if args.cond_drop is None else args.cond_drop
    elif exp == "E_full":
        cond_drop = 0.25 if args.cond_drop is None else args.cond_drop
    elif exp == "F_freq":
        cond_drop = 0.25 if args.cond_drop is None else args.cond_drop
        if args.freq_lambda == 0.0:
            args.freq_lambda = 0.1
    elif exp == "G_grad":
        cond_drop = 0.25 if args.cond_drop is None else args.cond_drop
        if args.grad_lambda == 0.0:
            args.grad_lambda = 0.05
    elif exp == "K_color":
        cond_drop = 0.25 if args.cond_drop is None else args.cond_drop
        if args.freq_lambda == 0.0:
            args.freq_lambda = 0.1
        if args.grad_lambda == 0.0:
            args.grad_lambda = 0.05
        if args.taware == 0:
            args.taware = 1
        args.radar_channels = 3
    else:  # H_taware
        cond_drop = 0.25 if args.cond_drop is None else args.cond_drop
        if args.freq_lambda == 0.0:
            args.freq_lambda = 0.1
        if args.grad_lambda == 0.0:
            args.grad_lambda = 0.05
        if args.taware == 0:
            args.taware = 1

    cfg = TrainConfig(
        exp=exp,
        root=args.root,
        img_size=args.img_size,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        run_name=args.run_name,
        cond_drop=cond_drop,
        use_film=use_film,
        use_cross_attn=use_cross_attn,
        use_amp=bool(args.use_amp),
        num_workers=args.num_workers,
        radar_channels=args.radar_channels,
        use_vae=bool(args.use_vae),
        latent_channels=args.latent_channels,
        vae_beta=args.vae_beta,
        vae_num_downsamples=args.vae_num_downsamples,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        enable_cache=bool(args.enable_cache),
        cache_in_workers=bool(args.cache_in_workers),
        preload_videos=bool(args.preload_videos),
        freq_lambda=args.freq_lambda,
        freq_band_split1=args.freq_band_split1,
        freq_band_split2=args.freq_band_split2,
        debug_freq=args.debug_freq,
        grad_lambda=args.grad_lambda,
        grad_mode=args.grad_mode,
        grad_on=args.grad_on,
        debug_grad=args.debug_grad,
        taware=args.taware,
        t_low=args.t_low,
        t_high=args.t_high,
        t_mix_power=args.t_mix_power,
        use_action_head=bool(args.use_action_head),
        action_head_dropout=args.action_head_dropout,
        action_head_dim=args.action_head_dim,
        action_adv=bool(args.action_adv),
        adv_lambda=args.adv_lambda,
        perc_lambda=args.perc_lambda,
        ssim_lambda=args.ssim_lambda,
        recon_lambda=args.recon_lambda,
        band_l1_lambda=args.band_l1_lambda,
        temporal_lambda=args.temporal_lambda,
        infonce_lambda=args.infonce_lambda,
        contrast_start_epoch=args.contrast_start_epoch,
        adv_start_epoch=args.adv_start_epoch,
        ema_decay=args.ema_decay,
        detect_nan=bool(args.detect_nan),
        video_encoder_type=args.video_encoder_type,
    )
    os.makedirs("outputs", exist_ok=True)
    run_training(cfg)


if __name__ == "__main__":
    main()
