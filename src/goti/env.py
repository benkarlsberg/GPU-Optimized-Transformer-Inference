from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _safe(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def _os_name() -> str:
    """Human-readable OS name. platform.platform() reports Windows 11 as
    "Windows-10" on older Python versions; Windows 11 builds start at 22000."""
    system = platform.system()
    if system == "Windows":
        try:
            build = int(platform.version().split(".")[-1])
        except ValueError:
            build = 0
        release = "11" if build >= 22000 else platform.release()
        return f"Windows {release} (build {build})"
    return f"{system} {platform.release()}"


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "os": _os_name(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["torch"]["device0"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": props.total_memory,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
    except Exception as exc:  # pragma: no cover
        info["torch"] = {"error": str(exc)}

    info["nvidia_smi"] = _safe(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    info["nvcc"] = _safe(["nvcc", "--version"])
    return info
