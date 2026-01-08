from __future__ import annotations

import json
import os
import random
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
from models.video2radar_baselines import Vid2VidGANDiscriminator, Vid2VidGANGenerator


@dataclass
class GanTrainConfig:
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
    base_channels: int = 64
    z_dim: int = 128
    cond_dim: int = 256
    video_encoder_type: str = "temporal_unet"
    adv_lambda: float = 1.0
    recon_lambda: float = 1.0


@dataclass
class GanSampleConfig:
    exp: str
    ckpt_path: str
    root: str
    split: str = "test"
    subject: str = "S10"
    img_size: int = 120
    clip_len: int = 64
    seed: int = 0
    run_name: Optional[str] = None
    radar_channels: int = 1
    num_per_class: int = 64


class GanTrainer:
    def __init__(self, cfg: GanTrainConfig) -> None:
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

        self.generator = Vid2VidGANGenerator(
            img_size=cfg.img_size,
            radar_channels=cfg.radar_channels,
            base_channels=cfg.base_channels,
            z_dim=cfg.z_dim,
            cond_dim=cfg.cond_dim,
            video_encoder_type=cfg.video_encoder_type,
        ).to(self.device)
        self.discriminator = Vid2VidGANDiscriminator(in_channels=cfg.radar_channels).to(self.device)
        self.opt_g = torch.optim.AdamW(self.generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_enabled = cfg.use_amp and scaler_device == "cuda"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=self.amp_enabled)
        self.best_val = float("inf")

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

    def _adv_loss(self, logits: torch.Tensor, target: float) -> torch.Tensor:
        labels = torch.full_like(logits, target)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def train_epoch(self) -> float:
        self.generator.train()
        self.discriminator.train()
        total = 0.0
        for batch in tqdm(self.train_loader, desc=f"[gan][train][{self.cfg.exp}]", leave=False):
            radar = batch["radar"].to(self.device).float()
            video = batch["video"].to(self.device)
            labels = batch["label"].to(self.device)

            self.opt_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                fake = self.generator(video, labels).detach()
                logits_real = self.discriminator(radar)
                logits_fake = self.discriminator(fake)
                loss_d = 0.5 * (self._adv_loss(logits_real, 1.0) + self._adv_loss(logits_fake, 0.0))
            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.opt_d)
            self.scaler.update()

            self.opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                fake = self.generator(video, labels)
                logits_fake = self.discriminator(fake)
                adv_loss = self._adv_loss(logits_fake, 1.0)
                recon_loss = F.l1_loss(fake, radar)
                loss_g = self.cfg.adv_lambda * adv_loss + self.cfg.recon_lambda * recon_loss
            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.opt_g)
            self.scaler.update()

            total += loss_g.item()
        return total / max(1, len(self.train_loader))

    def eval_epoch(self) -> float:
        self.generator.eval()
        total = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"[gan][val][{self.cfg.exp}]", leave=False):
                radar = batch["radar"].to(self.device).float()
                video = batch["video"].to(self.device)
                labels = batch["label"].to(self.device)
                fake = self.generator(video, labels)
                loss = F.l1_loss(fake, radar)
                total += loss.item()
        return total / max(1, len(self.val_loader))

    def run(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.eval_epoch()
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            print(f"[gan][{self.cfg.exp}] epoch={epoch}/{self.cfg.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            is_best = val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.ckpt")
            save_checkpoint(
                ckpt_path,
                self.generator,
                self.opt_g,
                epoch=epoch,
                step=epoch,
                config=asdict(self.cfg),
                best=is_best,
                metrics=metrics,
                extra_state={"discriminator": self.discriminator.state_dict()},
                save_last=True,
            )


def run_gan_training(cfg: GanTrainConfig) -> None:
    trainer = GanTrainer(cfg)
    trainer.run()


def run_gan_sampling(cfg: GanSampleConfig) -> None:
    from datasets.real_video_radar import ACTIONS, _default_video_transform, _load_video_clip
    from torchvision.utils import save_image

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
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
    generator = Vid2VidGANGenerator(
        img_size=model_cfg.get("img_size", cfg.img_size),
        radar_channels=model_cfg.get("radar_channels", cfg.radar_channels),
        base_channels=model_cfg.get("base_channels", 64),
        z_dim=model_cfg.get("z_dim", 128),
        cond_dim=model_cfg.get("cond_dim", 256),
        video_encoder_type=model_cfg.get("video_encoder_type", "temporal_unet"),
    ).to(device)
    generator.load_state_dict(state["model"])
    generator.eval()
    video_transform = _default_video_transform(cfg.img_size)

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
            rng = random.Random(cfg.seed + idx)
            clip = _load_video_clip(
                video_files[idx % len(video_files)],
                clip_len=cfg.clip_len,
                img_size=cfg.img_size,
                rng=rng,
                video_transform=video_transform,
            )
            clip = clip.unsqueeze(0).to(device)
            label = torch.tensor([ACTIONS.index(action)], device=device)
            with torch.no_grad():
                fake = generator(clip, label)
            img = (fake.clamp(-1, 1) + 1) / 2.0
            save_path = os.path.join(sample_dir, action, f"{idx:04d}.png")
            save_image(img, save_path)


__all__ = [
    "GanTrainConfig",
    "GanSampleConfig",
    "run_gan_training",
    "run_gan_sampling",
]
