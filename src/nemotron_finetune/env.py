"""Environment / hardware capture and GPU memory helpers.

Everything here is best-effort: a failure to read one field must never crash a
training run, so probes are wrapped defensively.
"""

from __future__ import annotations

import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict

_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "mamba_ssm",
    "causal_conv1d",
    "wandb",
    "omegaconf",
]


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"
    except Exception as exc:  # pragma: no cover - defensive
        return f"error:{exc}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_environment() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of the runtime environment."""
    info: Dict[str, Any] = {
        "timestamp": utcnow_iso(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _pkg_version(name) for name in _PACKAGES},
    }

    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1e9, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        info["cuda_devices"] = devices
    except Exception as exc:  # pragma: no cover - defensive
        info["torch_probe_error"] = str(exc)

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            info["nvidia_driver"] = out.stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover - defensive
        pass

    return info


def gpu_memory_stats(device: int = 0) -> Dict[str, float]:
    """Current / peak GPU memory in GiB for the given device (empty if no CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_mem_allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 3),
            "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved(device) / 1024**3, 3),
            "gpu_mem_max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        }
    except Exception:  # pragma: no cover - defensive
        return {}


def reset_peak_gpu_memory(device: int = 0) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # pragma: no cover - defensive
        pass
