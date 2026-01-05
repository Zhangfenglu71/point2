from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import torch
from torchvision.utils import save_image

from datasets.real_video_radar import ACTIONS, _default_video_transform, _load_video_clip
from models.unet import UNet
from models.video_encoder import SimpleVideoEncoder


def _build_schedule_fn(schedule: str, w: float, w0: float, w1: float) -> Callable[[float], float]:
    if schedule is None or schedule == "const":
        return lambda t: w
    if schedule == "linear":
        return lambda t: w0 + (w1 - w0) * t
    raise ValueError(f"Unknown schedule: {schedule}")


@dataclass
class SampleConfig:
    exp: str
    ckpt_path: str
    root: str
    split: str = "test"
    subject: str = "S10"
    img_size: int = 120
    clip_len: int = 64
    steps: int = 50
    seed: int = 0
    run_name: Optional[str] = None
    radar_channels: int = 1
    channel_mults: tuple[int, ...] = (1, 2, 4)
    cfg_w: float = 1.0
    cfg_w0: float = 1.0
    cfg_w1: float = 1.0
    schedule: str = "const"
    num_per_class: int = 64
    debug: bool = False
    debug_samples: int = 3


class Sampler:
    def __init__(self, cfg: SampleConfig) -> None:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        # Hard-code a stable default run name per experiment to avoid timestamped directories.
        default_name = f"sample_{cfg.exp}"
        self.run_name = cfg.run_name or default_name
        self.run_dir = os.path.join("outputs", "runs", self.run_name)
        self.sample_dir = os.path.join(self.run_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self._save_config()

        state = torch.load(cfg.ckpt_path, map_location=self.device)
        model_cfg = state.get("config", {})
        ckpt_exp = model_cfg.get("exp", cfg.exp)
        if ckpt_exp == "C_full":
            ckpt_exp = "E_full"
        # Backward compatibility: old D_full (with CFG/dropout) should map to E_full.
        if ckpt_exp == "D_full":
            cond_drop = model_cfg.get("cond_drop", 0.0)
            if cond_drop >= 0.2:
                ckpt_exp = "E_full"
        use_cond = ckpt_exp in {"B_cond", "C_film", "D_full", "E_full", "F_freq"}
        cond_dim = model_cfg.get("cond_dim", 256 if use_cond else None) if use_cond else None
        channel_mults = tuple(model_cfg.get("channel_mults", cfg.channel_mults))
        use_film = model_cfg.get("use_film", False) if use_cond else False
        use_cross_attn = model_cfg.get("use_cross_attn", False) if use_cond else False
        self.exp = ckpt_exp
        self.model = UNet(
            in_channels=model_cfg.get("radar_channels", cfg.radar_channels),
            cond_dim=cond_dim,
            use_film=use_film,
            use_cross_attn=use_cross_attn,
            cross_heads=model_cfg.get("cross_heads", 4),
            channel_mults=channel_mults,
        ).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        self.video_encoder: Optional[SimpleVideoEncoder]
        if use_cond:
            self.video_encoder = SimpleVideoEncoder(emb_dim=cond_dim).to(self.device)
            if "video_encoder" in state:
                self.video_encoder.load_state_dict(state["video_encoder"])
            self.video_encoder.eval()
        else:
            self.video_encoder = None

        self.video_transform = _default_video_transform(cfg.img_size)
        self.schedule_fn = _build_schedule_fn(cfg.schedule, cfg.cfg_w, cfg.cfg_w0, cfg.cfg_w1)

    def _save_config(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        try:
            status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
            git_state = "clean" if status == "" else "dirty"
        except Exception:
            git_state = "unavailable"
        cfg = asdict(self.cfg)
        cfg["git_state"] = git_state
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def _load_random_clip(self, video_files: list[str], idx: int) -> torch.Tensor:
        if not video_files:
            raise RuntimeError("No video files provided for sampling")
        video_path = video_files[idx % len(video_files)]
        rng = random.Random(self.cfg.seed + idx)
        clip = _load_video_clip(
            video_path,
            clip_len=self.cfg.clip_len,
            img_size=self.cfg.img_size,
            rng=rng,
            video_transform=self.video_transform,
        )
        return clip

    def _guided_velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.video_encoder is None or cond_emb is None or self.exp == "A_base":
            return self.model(x_t, t, None)
        if self.exp in {"C_film", "D_full"}:
            return self.model(x_t, t, cond_emb)
        with torch.no_grad():
            v_cond = self.model(x_t, t, cond_emb)
            v_uncond = self.model(x_t, t, torch.zeros_like(cond_emb))
            w = self.schedule_fn(float(t[0].item()))
            return v_uncond + w * (v_cond - v_uncond)

    def sample_action(self, action: str) -> None:
        os.makedirs(os.path.join(self.sample_dir, action), exist_ok=True)
        video_dir = os.path.join(self.cfg.root, self.cfg.split, "video", self.cfg.subject, action)
        if not os.path.isdir(video_dir):
            raise RuntimeError(f"Missing video directory: {video_dir}")
        video_files = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        if not video_files:
            raise RuntimeError(f"No videos found in {video_dir}")
        if self.cfg.debug:
            print(f"[sample][{action}] video_dir={video_dir}, num_videos={len(video_files)}")

        for idx in range(self.cfg.num_per_class):
            clip = self._load_random_clip(video_files, idx)
            clip = clip.unsqueeze(0).to(self.device)
            cond_emb = None
            if self.video_encoder is not None:
                with torch.no_grad():
                    cond_emb = self.video_encoder(clip)
                    if self.cfg.debug and idx < self.cfg.debug_samples:
                        emb = cond_emb.detach().cpu()
                        print(
                            f"[sample][{action}][idx={idx}] cond_emb mean={emb.mean().item():.4f} "
                            f"std={emb.std().item():.4f} first5={emb.view(-1)[:5].tolist()}"
                        )

            x = torch.randn((1, self.model.out_conv.out_channels, self.cfg.img_size, self.cfg.img_size), device=self.device)
            dt = 1.0 / float(max(1, self.cfg.steps))
            for step in range(self.cfg.steps):
                t_scalar = float(step) / float(max(1, self.cfg.steps))
                t_tensor = torch.full((1,), t_scalar, device=self.device)
                v = self._guided_velocity(x, t_tensor, cond_emb)
                x = x + dt * v
            img = (x.clamp(-1, 1) + 1) / 2.0
            save_path = os.path.join(self.sample_dir, action, f"{idx:04d}.png")
            save_image(img, save_path)

    def run(self) -> None:
        for action in ACTIONS:
            self.sample_action(action)


def run_sampling(cfg: SampleConfig) -> None:
    sampler = Sampler(cfg)
    sampler.run()


__all__ = ["SampleConfig", "run_sampling"]
