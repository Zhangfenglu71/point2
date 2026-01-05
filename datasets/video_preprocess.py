import colorsys
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F

decord.bridge.set_bridge("torch")

ColorMap = Literal["turbo", "hsl"]
SpectrogramType = Literal["stft", "chirp"]


@dataclass
class VideoPreprocessConfig:
    target_fps: float = 25.0
    clip_len: int = 64
    img_size: int = 224
    spectrogram_type: SpectrogramType = "stft"
    n_fft: int = 256
    hop_length: int = 64
    win_length: Optional[int] = None
    chirp_f0: float = 0.0
    chirp_f1: float = 0.5
    color_map: ColorMap = "turbo"
    min_clip_len: int = 48
    random_time_crop: bool = True
    speed_perturb_range: Tuple[float, float] = (0.9, 1.1)
    freq_noise_std: float = 0.01
    mask_prob: float = 0.35
    cutout_prob: float = 0.35
    mask_max_ratio: float = 0.2
    cutout_max_ratio: float = 0.25
    device: torch.device = torch.device("cpu")


class VideoSpectrogramPreprocessor:
    """Preprocess raw videos into colored spectrogram tensors with augmentations."""

    def __init__(self, config: VideoPreprocessConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self.rng = random.Random(seed)

    def __call__(self, video_path: str, training: bool = True) -> Dict[str, torch.Tensor]:
        frames = self._load_frames(video_path, training=training)
        frames = self._ensure_min_length(frames)
        clip = self._sample_clip(frames, training=training)
        spectrogram = self._frames_to_spectrogram(clip, training=training)
        colored = self._colorize_spectrogram(spectrogram)
        colored = F.interpolate(
            colored.unsqueeze(0),
            size=(self.config.img_size, self.config.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return {"spectrogram": colored, "frames": clip}

    def _load_frames(self, video_path: str, training: bool) -> torch.Tensor:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total == 0:
            raise ValueError(f"Video at {video_path} has no frames.")

        orig_fps = vr.get_avg_fps() or self.config.target_fps
        speed = self._sample_speed(training)
        effective_fps = max(1e-3, self.config.target_fps * speed)
        frame_step = max(1, int(round(orig_fps / effective_fps)))
        indices = list(range(0, total, frame_step))
        frames = vr.get_batch(indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W) in [0,1]
        return frames.to(self.config.device)

    def _sample_speed(self, training: bool) -> float:
        if not training:
            return 1.0
        low, high = self.config.speed_perturb_range
        if low == high:
            return low
        return self.rng.uniform(low, high)

    def _ensure_min_length(self, frames: torch.Tensor) -> torch.Tensor:
        min_len = max(self.config.clip_len, self.config.min_clip_len)
        if frames.size(0) >= min_len:
            return frames

        # Mirror padding first, then loop if still short.
        mirrored: List[torch.Tensor] = []
        while len(mirrored) + frames.size(0) < min_len:
            mirrored.append(frames.flip(0))
            frames = torch.cat([frames, mirrored[-1]], dim=0)
        if frames.size(0) < min_len:
            repeat_count = math.ceil(min_len / frames.size(0))
            frames = frames.repeat((repeat_count, 1, 1, 1))[:min_len]
        return frames

    def _sample_clip(self, frames: torch.Tensor, training: bool) -> torch.Tensor:
        target_len = max(self.config.clip_len, self.config.min_clip_len)
        total = frames.size(0)
        if total <= target_len:
            return frames[:target_len]

        if training and self.config.random_time_crop:
            start = self.rng.randint(0, total - target_len)
        else:
            start = max(0, (total - target_len) // 2)
        end = start + target_len
        return frames[start:end]

    def _frames_to_spectrogram(self, frames: torch.Tensor, training: bool) -> torch.Tensor:
        grayscale = (
            0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        )  # (T, H, W)
        signal = grayscale.mean(dim=2).transpose(0, 1)  # (H, T)
        if self.config.spectrogram_type == "chirp":
            signal = self._apply_chirp(signal)
        win_length = self.config.win_length or self.config.n_fft
        spec = torch.stft(
            signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=win_length,
            return_complex=True,
        )  # (H, freq, time)
        magnitude = spec.abs().mean(dim=0)  # average over height -> (freq, time)
        magnitude = torch.log1p(magnitude)
        magnitude = self._apply_frequency_noise(magnitude, training=training)
        magnitude = self._apply_masking(magnitude, training=training)
        magnitude = self._normalize(magnitude)
        return magnitude

    def _apply_chirp(self, signal: torch.Tensor) -> torch.Tensor:
        t = torch.linspace(0, 1, signal.size(-1), device=signal.device)
        phase = 2 * math.pi * (self.config.chirp_f0 * t + 0.5 * (self.config.chirp_f1 - self.config.chirp_f0) * t * t)
        chirp = torch.cos(phase)
        return signal * chirp

    def _apply_frequency_noise(self, magnitude: torch.Tensor, training: bool) -> torch.Tensor:
        if not training or self.config.freq_noise_std <= 0:
            return magnitude
        noise = torch.randn_like(magnitude) * self.config.freq_noise_std
        return magnitude + noise

    def _apply_masking(self, magnitude: torch.Tensor, training: bool) -> torch.Tensor:
        if not training:
            return magnitude

        freq_bins, time_bins = magnitude.shape
        mag = magnitude.clone()

        if self.rng.random() < self.config.mask_prob:
            max_h = max(1, int(freq_bins * self.config.mask_max_ratio))
            max_w = max(1, int(time_bins * self.config.mask_max_ratio))
            h = self.rng.randint(1, max_h)
            w = self.rng.randint(1, max_w)
            top = self.rng.randint(0, max(0, freq_bins - h))
            left = self.rng.randint(0, max(0, time_bins - w))
            mag[top : top + h, left : left + w] = mag.mean()

        if self.rng.random() < self.config.cutout_prob:
            max_h = max(1, int(freq_bins * self.config.cutout_max_ratio))
            max_w = max(1, int(time_bins * self.config.cutout_max_ratio))
            h = self.rng.randint(1, max_h)
            w = self.rng.randint(1, max_w)
            top = self.rng.randint(0, max(0, freq_bins - h))
            left = self.rng.randint(0, max(0, time_bins - w))
            mag[top : top + h, left : left + w] = 0.0

        return mag

    def _normalize(self, magnitude: torch.Tensor) -> torch.Tensor:
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        if mag_max <= mag_min:
            return torch.zeros_like(magnitude)
        return (magnitude - mag_min) / (mag_max - mag_min + 1e-6)

    def _colorize_spectrogram(self, magnitude: torch.Tensor) -> torch.Tensor:
        mag_np = magnitude.clamp(0, 1).cpu().numpy()
        if self.config.color_map == "turbo":
            colored = self._apply_turbo(mag_np)
        else:
            colored = self._apply_hsl(mag_np)
        tensor = torch.from_numpy(colored).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.config.device)

    def _apply_turbo(self, mag_np: np.ndarray) -> np.ndarray:
        uint8_img = (mag_np * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(uint8_img, cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _apply_hsl(self, mag_np: np.ndarray) -> np.ndarray:
        hue = mag_np  # 0-1
        saturation = np.clip(np.sqrt(mag_np), 0, 1)
        lightness = np.clip(0.45 + 0.35 * mag_np, 0, 1)
        flat_h = hue.flatten()
        flat_s = saturation.flatten()
        flat_l = lightness.flatten()

        def _to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            return r, g, b

        rgb_flat = np.array([_to_rgb(h, s, l) for h, s, l in zip(flat_h, flat_s, flat_l)], dtype=np.float32)
        rgb = rgb_flat.reshape((*mag_np.shape, 3))
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
