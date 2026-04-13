"""Torch runtime configuration helpers for faster inference."""

from __future__ import annotations

import os
from threading import Lock

import torch


_RUNTIME_CONFIGURED = False
_RUNTIME_LOCK = Lock()


def configure_torch_runtime() -> None:
    """Apply one-time Torch settings that favor inference performance."""
    global _RUNTIME_CONFIGURED

    if _RUNTIME_CONFIGURED:
        return

    with _RUNTIME_LOCK:
        if _RUNTIME_CONFIGURED:
            return

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True

            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        else:
            cpu_count = os.cpu_count() or 1
            num_threads = max(1, min(cpu_count, 8))

            try:
                torch.set_num_threads(num_threads)
            except RuntimeError:
                pass

            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        _RUNTIME_CONFIGURED = True