"""Reproducibility utilities for deterministic experiment execution."""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, received: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Also set python hash seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def get_device(requested_device: Optional[str] = "auto") -> torch.device:
    """Selects and returns an available PyTorch device ('cuda' or 'cpu')."""
    if requested_device is None or requested_device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_str = requested_device.lower().strip()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(device_str)
