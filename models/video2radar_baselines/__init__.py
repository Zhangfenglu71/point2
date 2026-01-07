"""Baseline video-to-radar models for comparison experiments."""

from models.video2radar_baselines.gan_vid2vid import Vid2VidGANGenerator, Vid2VidGANDiscriminator
from models.video2radar_baselines.diffusion_3d_unet import VideoDiffusion3DUNet
from models.video2radar_baselines.diffusion_st_attn import VideoDiffusionSTAttn
from models.video2radar_baselines.diffusion_attn_control import VideoDiffusionAttnControl
from models.video2radar_baselines.diffusion_segmented_attn import VideoDiffusionSegmentedAttn

__all__ = [
    "Vid2VidGANGenerator",
    "Vid2VidGANDiscriminator",
    "VideoDiffusion3DUNet",
    "VideoDiffusionSTAttn",
    "VideoDiffusionAttnControl",
    "VideoDiffusionSegmentedAttn",
]
