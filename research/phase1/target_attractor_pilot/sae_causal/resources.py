from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WindowsCommitMemory:
    committed_gb: float
    limit_gb: float
    headroom_gb: float


def windows_commit_memory() -> WindowsCommitMemory | None:
    if os.name != "nt":
        return None

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong),
        ]

    value = PerformanceInformation()
    value.cb = ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value), value.cb):
        return None
    scale = value.PageSize / (1024 ** 3)
    committed, limit = value.CommitTotal * scale, value.CommitLimit * scale
    return WindowsCommitMemory(committed, limit, limit - committed)


def require_commit_headroom(minimum_gb: float) -> WindowsCommitMemory | None:
    status = windows_commit_memory()
    if status is not None and status.headroom_gb < minimum_gb:
        raise RuntimeError(
            f"insufficient Windows commit headroom for Qwen3-8B loading: "
            f"{status.headroom_gb:.2f} GB available, {minimum_gb:.2f} GB required; "
            "close memory-heavy applications or reboot, then rerun the same command"
        )
    return status
