import os
import random
from dataclasses import dataclass
from glob import glob
from typing import Callable, Dict, List, Optional, Tuple

import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


decord.bridge.set_bridge("torch")

ACTIONS = ["box", "jump", "run", "walk"]


def _default_image_transform(img_size: int, radar_channels: int) -> transforms.Compose:
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((img_size, img_size))
    normalize = transforms.Normalize([0.5] * radar_channels, [0.5] * radar_channels)
    return transforms.Compose([resize, to_tensor, normalize])


def _default_video_transform(img_size: int) -> transforms.Compose:
    def transform(frames: torch.Tensor) -> torch.Tensor:
        # frames: (T, C, H, W) in [0,1]
        frames = torch.nn.functional.interpolate(
            frames, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
        frames = (frames - 0.5) / 0.5
        return frames

    return transforms.Compose([transform])


def _list_subject_paths(root: str, split: str, modality: str) -> List[str]:
    base = os.path.join(root, split, modality)
    subjects = sorted([p for p in glob(os.path.join(base, "S*")) if os.path.isdir(p)])
    return subjects


def _load_image(path: str, transform: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
    with Image.open(path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        tensor = transform(img)
    return tensor


def _load_video_clip(
    path: str,
    clip_len: int,
    img_size: int,
    rng: random.Random,
    video_transform: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    vr = decord.VideoReader(path)
    total = len(vr)
    if total <= clip_len:
        indices = list(range(total))
        while len(indices) < clip_len:
            indices.append(indices[-1])
    else:
        start = rng.randint(0, total - clip_len)
        indices = list(range(start, start + clip_len))
    frames = vr.get_batch(indices)  # (T, H, W, C) torch tensor due to bridge
    frames = frames.permute(0, 3, 1, 2) / 255.0
    frames = video_transform(frames)
    return frames


def _sample_clip_from_frames(
    frames: torch.Tensor, clip_len: int, rng: random.Random
) -> torch.Tensor:
    """Sample a temporal clip from predecoded frames (T, C, H, W)."""
    total = frames.size(0)
    if total <= clip_len:
        indices = list(range(total))
        while len(indices) < clip_len:
            indices.append(indices[-1])
    else:
        start = rng.randint(0, total - clip_len)
        indices = list(range(start, start + clip_len))
    return frames[indices]


@dataclass
class RealVideoRadarConfig:
    root: str
    split: str
    img_size: int = 120
    clip_len: int = 64
    radar_channels: int = 1


class RealVideoRadarDataset(Dataset):
    """Aligned radar spectrogram and video clip dataset."""

    def __init__(
        self,
        config: RealVideoRadarConfig,
        seed: Optional[int] = None,
        radar_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        video_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        enable_cache: bool = True,
        preload_videos: bool = False,
    ) -> None:
        self.config = config
        self.root = config.root
        self.split = config.split
        self.img_size = config.img_size
        self.clip_len = config.clip_len
        self.radar_channels = config.radar_channels
        self.rng = random.Random(seed)
        self.enable_cache = enable_cache
        self.preload_videos = preload_videos

        self.video_cache: Dict[str, torch.Tensor] = {}

        self.radar_transform = radar_transform or _default_image_transform(
            img_size=self.img_size, radar_channels=self.radar_channels
        )
        self.video_transform = video_transform or _default_video_transform(img_size=self.img_size)

        self.samples: List[Tuple[str, str, str]] = []
        radar_subjects = _list_subject_paths(self.root, self.split, "radar")
        for subject_path in radar_subjects:
            subject = os.path.basename(subject_path)
            for action in ACTIONS:
                pattern = os.path.join(subject_path, action, "*")
                for radar_path in sorted(glob(pattern)):
                    if not os.path.isfile(radar_path):
                        continue
                    # store subject, action, radar_path
                    self.samples.append((subject, action, radar_path))

        # index available videos per (subject, action)
        self.videos: Dict[Tuple[str, str], List[str]] = {}
        video_subjects = _list_subject_paths(self.root, self.split, "video")
        for subject_path in video_subjects:
            subject = os.path.basename(subject_path)
            for action in ACTIONS:
                pattern = os.path.join(subject_path, action, "*")
                candidates = [p for p in sorted(glob(pattern)) if os.path.isfile(p)]
                if candidates:
                    self.videos[(subject, action)] = candidates

        if not self.samples:
            raise ValueError(f"No radar samples found under {self.root}/{self.split}")

        if self.preload_videos:
            self.preload_all_videos()

    def __len__(self) -> int:
        return len(self.samples)

    def preload_all_videos(self) -> None:
        """Decode and cache all videos for this split upfront."""
        if not self.enable_cache:
            print(f"[RealVideoRadarDataset] Cache disabled; skipping preload for split={self.split}.")
            return

        seen = 0
        for key, video_paths in self.videos.items():
            subject, action = key
            for video_path in video_paths:
                cache_key = str(video_path)
                if cache_key in self.video_cache:
                    continue
                _ = self._get_video_frames(video_path)
                seen += 1
        print(f"[RealVideoRadarDataset] Preloaded {seen} videos into cache for split={self.split}.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject, action, radar_path = self.samples[idx]
        radar = _load_image(radar_path, self.radar_transform)
        if self.radar_channels == 1 and radar.shape[0] == 1:
            pass
        elif self.radar_channels == 1 and radar.shape[0] == 3:
            radar = radar[:1]
        elif self.radar_channels == 3 and radar.shape[0] == 1:
            radar = radar.repeat(3, 1, 1)

        key = (subject, action)
        if key not in self.videos:
            raise RuntimeError(f"No video found for {key}")
        video_path = self.rng.choice(self.videos[key])
        video_clip = self._get_video_clip(video_path)

        label = torch.tensor(ACTIONS.index(action), dtype=torch.long)
        return {
            "radar": radar,
            "video": video_clip,
            "label": label,
            "subject": subject,
            "radar_path": radar_path,
            "video_path": video_path,
        }

    def _decode_full_video(self, video_path: str) -> torch.Tensor:
        """Decode an entire video and apply transforms once."""
        vr = decord.VideoReader(video_path)
        frames = vr.get_batch(list(range(len(vr))))  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2) / 255.0  # (T, C, H, W) in [0,1]
        frames = self.video_transform(frames)  # resize + normalize
        return frames

    def _get_video_frames(self, video_path: str) -> torch.Tensor:
        cache_key = str(video_path)
        if self.enable_cache and cache_key in self.video_cache:
            return self.video_cache[cache_key]

        frames = self._decode_full_video(video_path)
        if self.enable_cache:
            self.video_cache[cache_key] = frames
        return frames

    def _get_video_clip(self, video_path: str) -> torch.Tensor:
        if self.enable_cache:
            frames = self._get_video_frames(video_path)
            return _sample_clip_from_frames(frames, self.clip_len, self.rng)
        return _load_video_clip(
            video_path,
            clip_len=self.clip_len,
            img_size=self.img_size,
            rng=self.rng,
            video_transform=self.video_transform,
        )
