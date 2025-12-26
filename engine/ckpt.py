import json
import os
from typing import Any, Dict, Tuple

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: Dict[str, Any],
    best: bool = False,
    metrics: Dict[str, Any] | None = None,
    extra_state: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
        "metrics": metrics or {},
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)
    meta = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
    }
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if best:
        best_path = os.path.join(os.path.dirname(path), "best.ckpt")
        torch.save(state, best_path)


def load_checkpoint(path: str, map_location: str | None = None) -> Tuple[Dict[str, Any], Any]:
    state = torch.load(path, map_location=map_location)
    config = state.get("config", {})
    return state, config
