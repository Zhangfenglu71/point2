from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class UNetConditioning:
    """Container for UNet conditioning vectors and per-scale tokens."""

    vector: Optional[torch.Tensor] = None
    scale_tokens: Optional[List[torch.Tensor]] = None

    def apply_dropout(self, mask: torch.Tensor) -> "UNetConditioning":
        """Apply classifier-free guidance dropout mask to all components."""
        new_vector = self.vector * mask if self.vector is not None else None
        if self.scale_tokens is None:
            new_tokens = None
        else:
            new_tokens = [tok * mask.view(-1, 1, 1) for tok in self.scale_tokens]
        return UNetConditioning(vector=new_vector, scale_tokens=new_tokens)

    def zeros_like(self) -> "UNetConditioning":
        """Return a zero-conditioning counterpart for CFG sampling."""
        zero_vector = torch.zeros_like(self.vector) if self.vector is not None else None
        zero_tokens = None
        if self.scale_tokens is not None:
            zero_tokens = [torch.zeros_like(tok) for tok in self.scale_tokens]
        return UNetConditioning(vector=zero_vector, scale_tokens=zero_tokens)
