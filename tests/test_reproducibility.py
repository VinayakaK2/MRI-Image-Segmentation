"""Tests for seed setting, deterministic behavior, and device selection."""

import pytest
import numpy as np
import torch
from src.utils.reproducibility import set_seed, get_device


def test_set_seed_returns_seed() -> None:
    """Verifies that set_seed returns the configured integer seed."""
    seed_val = 1337
    result = set_seed(seed_val)
    assert result == seed_val


def test_set_seed_invalid_raises_error() -> None:
    """Verifies that a negative seed raises ValueError."""
    with pytest.raises(ValueError):
        set_seed(-5)


def test_numpy_and_torch_deterministic_outputs() -> None:
    """Verifies that identical seeds produce identical random sequences in NumPy and PyTorch."""
    set_seed(42)
    np_sample_1 = np.random.rand(5)
    torch_sample_1 = torch.rand(5)

    set_seed(42)
    np_sample_2 = np.random.rand(5)
    torch_sample_2 = torch.rand(5)

    assert np.allclose(np_sample_1, np_sample_2)
    assert torch.allclose(torch_sample_1, torch_sample_2)


def test_device_selection_cpu() -> None:
    """Verifies explicit CPU selection returns cpu device."""
    dev = get_device("cpu")
    assert dev.type == "cpu"


def test_device_selection_auto() -> None:
    """Verifies auto selection resolves to cuda if available else cpu."""
    dev = get_device("auto")
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected_type


def test_device_selection_cuda_fallback_on_cpu_only() -> None:
    """Verifies that requesting cuda falls back to cpu gracefully when no GPU exists."""
    if not torch.cuda.is_available():
        dev = get_device("cuda")
        assert dev.type == "cpu"
