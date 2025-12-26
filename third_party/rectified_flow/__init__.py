"""Lightweight rectified flow utilities.

This module provides a minimal scheduler and helper functions to keep the code
self contained. The implementation here is intentionally simple and does not
mirror any particular external repository; it only exposes what the rest of the
project needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass
class FlowSchedule:
    steps: int = 50

    def time_at(self, idx: int) -> float:
        return float(idx) / float(max(1, self.steps))


def euler_sampler(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise_shape: Tuple[int, ...],
    device: torch.device,
    steps: int,
) -> torch.Tensor:
    """Simple Euler solver for the straight-line rectified flow path.

    Args:
        model_fn: function mapping (x_t, t_tensor) -> velocity prediction
        noise_shape: shape for the base noise
        device: torch device
        steps: integration steps
    Returns:
        Generated sample tensor on the given device.
    """
    schedule = FlowSchedule(steps=steps)
    x = torch.randn(noise_shape, device=device)
    dt = 1.0 / float(max(1, steps))
    for i in range(steps):
        t = torch.full((noise_shape[0],), schedule.time_at(i), device=device)
        v = model_fn(x, t)
        x = x + dt * v
    return x
