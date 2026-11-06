"""Optional Windows adapter. Never retrieves titles, paths, contents or input."""

import sys

import psutil

from contextedge.context.mappings import categorize


def foreground_category(mappings: dict[str, str]) -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        hwnd = user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return categorize(psutil.Process(pid.value).name(), mappings) if pid.value else None
    except (OSError, psutil.Error):
        return None
