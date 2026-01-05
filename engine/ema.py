from __future__ import annotations

from typing import Dict, Optional

import torch


class ExponentialMovingAverage:
    """Simple EMA wrapper that tracks parameters and buffers.

    The helper keeps a shadow copy of trainable parameters (and buffers) and can
    swap them in for evaluation before restoring the original weights.
    """

    def __init__(self, module: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}
        self.backup_buffers: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def _init_from(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow_params[name] = param.detach().clone()
        for name, buf in module.named_buffers():
            self.shadow_buffers[name] = buf.detach().clone()
        self._initialized = True

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        if not self._initialized:
            self._init_from(module)
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            shadow = self.shadow_params[name]
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for name, buf in module.named_buffers():
            shadow_buf = self.shadow_buffers.get(name)
            if shadow_buf is None:
                self.shadow_buffers[name] = buf.detach().clone()
                continue
            shadow_buf.mul_(self.decay).add_(buf.detach(), alpha=1.0 - self.decay)

    def apply_to(self, module: torch.nn.Module) -> None:
        if not self._initialized:
            return
        self.backup_params = {}
        self.backup_buffers = {}
        for name, param in module.named_parameters():
            if name not in self.shadow_params:
                continue
            self.backup_params[name] = param.data.clone()
            param.data.copy_(self.shadow_params[name])
        for name, buf in module.named_buffers():
            if name not in self.shadow_buffers:
                continue
            self.backup_buffers[name] = buf.data.clone()
            buf.data.copy_(self.shadow_buffers[name])

    def restore(self, module: torch.nn.Module) -> None:
        if not self.backup_params and not self.backup_buffers:
            return
        for name, param in module.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        for name, buf in module.named_buffers():
            if name in self.backup_buffers:
                buf.data.copy_(self.backup_buffers[name])
        self.backup_params = {}
        self.backup_buffers = {}

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "shadow_params": self.shadow_params,
            "shadow_buffers": self.shadow_buffers,
        }

    def load_state_dict(self, state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> None:
        if state is None:
            return
        self.shadow_params = state.get("shadow_params", {})
        self.shadow_buffers = state.get("shadow_buffers", {})
        self._initialized = bool(self.shadow_params or self.shadow_buffers)
