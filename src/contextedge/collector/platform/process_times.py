"""Bounded-cost process CPU access. Never requests paths, contents or inputs."""

from __future__ import annotations

import sys

import psutil

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel.OpenProcess.restype = wintypes.HANDLE
    _kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _kernel.GetProcessTimes.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]


def process_times(process) -> tuple[float, float] | None:
    if process.pid == 0:
        return None
    if sys.platform != "win32":
        try:
            value = process.cpu_times()
            return process.create_time(), value.user + value.system
        except (psutil.Error, OSError):
            return None
    # psutil's Windows cpu_times fallback enumerates the whole process table for
    # EACH inaccessible PID. Skip inaccessible counters instead of paying O(n²).
    handle = _kernel.OpenProcess(0x1000, False, process.pid)  # QUERY_LIMITED_INFORMATION only
    if not handle:
        return None
    try:
        created, exited, kernel, user = [wintypes.FILETIME() for _ in range(4)]
        if not _kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in (created, exited, kernel, user))):
            return None

        def seconds(value):
            return ((value.dwHighDateTime << 32) | value.dwLowDateTime) / 10_000_000

        return seconds(created), seconds(kernel) + seconds(user)
    finally:
        _kernel.CloseHandle(handle)
