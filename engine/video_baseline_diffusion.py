from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.real_video_radar import RealVideoRadarConfig, RealVideoRadarDataset
from engine.ckpt import save_checkpoint
from engine.seed import seed_all
from models.video2radar_baselines import (
    VideoDiffusion3DUNet,
    VideoDiffusionAttnControl,
    VideoDiffusionSTAttn,
    VideoDiffusionSegmentedAttn,
)


@dataclass
class DiffusionTrainConfig:
    exp: str
    root: str
    split_train: str = "train"
    split_val: str = "val"
    img_size: int = 120
    clip_len: int = 64
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 0
    run_name: Optional[str] = None
    use_amp: bool = True
    radar_channels: int = 1
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cond_dim: int = 256
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    cross_heads: int = 4
    video_encoder_type: str = "temporal_unet"
    num_segments: int = 4


@dataclass
class DiffusionSampleConfig:
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
    num_per_class: int = 64
    video_encoder_type: str = "temporal_unet"


class DiffusionTrainer:
    def __init__(self, cfg: DiffusionTrainConfig) -> None:
        seed_all(cfg.seed)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = cfg.run_name or f"train_{cfg.exp}"
        self.run_dir = os.path.join("outputs", "runs", self.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._save_config()

        train_dataset = RealVideoRadarDataset(
            RealVideoRadarConfig(
                root=cfg.root,
                split=cfg.split_train,
                img_size=cfg.img_size,
                clip_len=cfg.clip_len,
                radar_channels=cfg.radar_channels,
            ),
            seed=cfg.seed,
        )
        val_dataset = RealVideoRadarDataset(
            RealVideoRadarConfig(
                root=cfg.root,
                split=cfg.split_val,
                img_size=cfg.img_size,
                clip_len=cfg.clip_len,
                radar_channels=cfg.radar_channels,
            ),
            seed=cfg.seed + 1,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.best_val = float("inf")

        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_enabled = cfg.use_amp and scaler_device == "cuda"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=self.amp_enabled)

    def _save_config(self) -> None:
        try:
            status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
            git_state = "clean" if status == "" else "dirty"
        except Exception:
            git_state = "unavailable"
        cfg_dict = asdict(self.cfg)
        cfg_dict["git_state"] = git_state
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    def _build_model(self) -> torch.nn.Module:
        if self.cfg.exp == "DIFF_3DUNet":
            return VideoDiffusion3DUNet(
                in_channels=self.cfg.radar_channels,
                base_channels=max(16, self.cfg.base_channels // 2),
                channel_mults=(1, 2, 4),
            )
        if self.cfg.exp == "DIFF_STAttn":
            return VideoDiffusionSTAttn(
                in_channels=self.cfg.radar_channels,
                base_channels=self.cfg.base_channels,
                channel_mults=self.cfg.channel_mults,
                cond_dim=self.cfg.cond_dim,
                video_encoder_type=self.cfg.video_encoder_type,
                cross_heads=self.cfg.cross_heads,
            )
        if self.cfg.exp == "DIFF_AttnCtrl":
            return VideoDiffusionAttnControl(
                in_channels=self.cfg.radar_channels,
                base_channels=self.cfg.base_channels,
                channel_mults=self.cfg.channel_mults,
                cond_dim=self.cfg.cond_dim,
                video_encoder_type=self.cfg.video_encoder_type,
                cross_heads=self.cfg.cross_heads,
            )
        if self.cfg.exp == "DIFF_SegAttn":
            return VideoDiffusionSegmentedAttn(
                in_channels=self.cfg.radar_channels,
                base_channels=self.cfg.base_channels,
                channel_mults=self.cfg.channel_mults,
                cond_dim=self.cfg.cond_dim,
                video_encoder_type=self.cfg.video_encoder_type,
                cross_heads=self.cfg.cross_heads,
                num_segments=self.cfg.num_segments,
            )
        raise ValueError(f"Unsupported diffusion exp: {self.cfg.exp}")

    def _predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        video = batch["video"].to(self.device)
        if self.cfg.exp == "DIFF_3DUNet":
            return self.model(x_t, t, video)
        labels = batch["label"].to(self.device)
        return self.model(x_t, t, video, labels)

    def _step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = batch["radar"].to(self.device).float()
        b = x0.size(0)
        t = torch.randint(0, self.cfg.diffusion_steps, (b,), device=self.device)
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(b, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
        with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            pred = self._predict_noise(x_t, t, batch)
            loss = F.mse_loss(pred, noise)
        return loss

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(self.train_loader, desc=f"[diff][train][{self.cfg.exp}]", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step(batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
        return total / max(1, len(self.train_loader))

    def eval_epoch(self) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"[diff][val][{self.cfg.exp}]", leave=False):
                loss = self._step(batch)
                total += loss.item()
        return total / max(1, len(self.val_loader))

    def run(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.eval_epoch()
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            print(
                f"[diff][{self.cfg.exp}] epoch={epoch}/{self.cfg.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )
            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.ckpt")
            save_checkpoint(
                ckpt_path,
                self.model,
                self.optimizer,
                epoch=epoch,
                step=epoch,
                config=asdict(self.cfg),
                best=is_best,
                metrics=metrics,
                save_last=True,
            )


def run_diffusion_training(cfg: DiffusionTrainConfig) -> None:
    trainer = DiffusionTrainer(cfg)
    trainer.run()


def _infer_3dunet_base_channels(state: Dict[str, Any]) -> Optional[int]:
    model_state = state.get("model", {})
    weight = model_state.get("enc_blocks.0.conv1.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() >= 1:
        return int(weight.size(0))
    weight = model_state.get("dec_blocks.0.conv1.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() >= 2:
        return int(weight.size(1) // 8)
    return None


def _load_model_from_ckpt(cfg: DiffusionSampleConfig, state: Dict[str, Any]) -> torch.nn.Module:
    exp = cfg.exp
    model_cfg = state.get("config", {})
    if exp == "DIFF_3DUNet":
        inferred_base = _infer_3dunet_base_channels(state)
        base_channels = inferred_base if inferred_base is not None else max(16, model_cfg.get("base_channels", 64) // 2)
        model = VideoDiffusion3DUNet(
            in_channels=model_cfg.get("radar_channels", cfg.radar_channels),
            base_channels=base_channels,
            channel_mults=(1, 2, 4),
        )
    elif exp == "DIFF_STAttn":
        model = VideoDiffusionSTAttn(
            in_channels=model_cfg.get("radar_channels", cfg.radar_channels),
            base_channels=model_cfg.get("base_channels", 64),
            channel_mults=tuple(model_cfg.get("channel_mults", (1, 2, 4))),
            cond_dim=model_cfg.get("cond_dim", 256),
            video_encoder_type=model_cfg.get("video_encoder_type", cfg.video_encoder_type),
            cross_heads=model_cfg.get("cross_heads", 4),
        )
    elif exp == "DIFF_AttnCtrl":
        model = VideoDiffusionAttnControl(
            in_channels=model_cfg.get("radar_channels", cfg.radar_channels),
            base_channels=model_cfg.get("base_channels", 64),
            channel_mults=tuple(model_cfg.get("channel_mults", (1, 2, 4))),
            cond_dim=model_cfg.get("cond_dim", 256),
            video_encoder_type=model_cfg.get("video_encoder_type", cfg.video_encoder_type),
            cross_heads=model_cfg.get("cross_heads", 4),
        )
    elif exp == "DIFF_SegAttn":
        model = VideoDiffusionSegmentedAttn(
            in_channels=model_cfg.get("radar_channels", cfg.radar_channels),
            base_channels=model_cfg.get("base_channels", 64),
            channel_mults=tuple(model_cfg.get("channel_mults", (1, 2, 4))),
            cond_dim=model_cfg.get("cond_dim", 256),
            video_encoder_type=model_cfg.get("video_encoder_type", cfg.video_encoder_type),
            cross_heads=model_cfg.get("cross_heads", 4),
            num_segments=model_cfg.get("num_segments", 4),
        )
    else:
        raise ValueError(f"Unsupported diffusion exp: {exp}")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _build_linear_schedule(steps: int, beta_start: float, beta_end: float, device: torch.device) -> tuple[torch.Tensor, ...]:
    betas = torch.linspace(beta_start, beta_end, steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def run_diffusion_sampling(cfg: DiffusionSampleConfig) -> None:
    import random

    from datasets.real_video_radar import ACTIONS, _default_video_transform, _load_video_clip
    from torchvision.utils import save_image

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.run_name or f"sample_{cfg.exp}"
    run_dir = os.path.join("outputs", "runs", run_name)
    sample_dir = os.path.join(run_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    try:
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        git_state = "clean" if status == "" else "dirty"
    except Exception:
        git_state = "unavailable"
    cfg_dict = asdict(cfg)
    cfg_dict["git_state"] = git_state
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    state = torch.load(cfg.ckpt_path, map_location=device)
    model_cfg = state.get("config", {})
    total_steps = int(model_cfg.get("diffusion_steps", 1000))
    betas, alphas, alpha_bars = _build_linear_schedule(
        total_steps,
        float(model_cfg.get("beta_start", 1e-4)),
        float(model_cfg.get("beta_end", 0.02)),
        device,
    )
    model = _load_model_from_ckpt(cfg, state).to(device)
    video_transform = _default_video_transform(cfg.img_size)

    def load_clip(video_files: list[str], idx: int) -> torch.Tensor:
        rng = random.Random(cfg.seed + idx)
        clip = _load_video_clip(
            video_files[idx % len(video_files)],
            clip_len=cfg.clip_len,
            img_size=cfg.img_size,
            rng=rng,
            video_transform=video_transform,
        )
        return clip

    for action in ACTIONS:
        os.makedirs(os.path.join(sample_dir, action), exist_ok=True)
        video_dir = os.path.join(cfg.root, cfg.split, "video", cfg.subject, action)
        video_files = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        if not video_files:
            raise RuntimeError(f"No videos found in {video_dir}")
        for idx in range(cfg.num_per_class):
            clip = load_clip(video_files, idx).unsqueeze(0).to(device)
            label = torch.tensor([ACTIONS.index(action)], device=device)
            x = torch.randn((1, cfg.radar_channels, cfg.img_size, cfg.img_size), device=device)
            sample_steps = max(1, min(cfg.steps, total_steps))
            indices = torch.linspace(0, total_steps - 1, sample_steps, device=device).long().tolist()
            for step in reversed(indices):
                t = torch.full((1,), step, device=device, dtype=torch.long)
                if cfg.exp == "DIFF_3DUNet":
                    eps = model(x, t, clip)
                else:
                    eps = model(x, t, clip, label)
                alpha = alphas[step]
                alpha_bar = alpha_bars[step]
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                mean = coef1 * (x - coef2 * eps)
                if step > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(betas[step])
                    x = mean + sigma * noise
                else:
                    x = mean
            img = (x.clamp(-1, 1) + 1) / 2.0
            save_path = os.path.join(sample_dir, action, f"{idx:04d}.png")
            save_image(img, save_path)


__all__ = [
    "DiffusionTrainConfig",
    "DiffusionSampleConfig",
    "run_diffusion_training",
    "run_diffusion_sampling",
]
