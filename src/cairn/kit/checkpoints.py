"""Atomic checkpoint writing and loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"checkpoint at {path} is not a state mapping")
    return blob
