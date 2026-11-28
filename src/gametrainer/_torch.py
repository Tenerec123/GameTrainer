"""Lazy, centralized import of the optional ``torch`` dependency.

``import gametrainer`` must never require torch. Every module that needs torch
imports the names from here instead of repeating the ``try/except`` dance.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    torch = None
    nn = None
    F = None
    Dataset = None
    DataLoader = None
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "This feature requires torch. Install with: pip install 'gametrainer[nn]'"
        )
