"""Root conftest — ensures libvips is discoverable on macOS.

On macOS with Homebrew, libvips lives in /opt/homebrew/lib which may
not be on the default library search path. This conftest sets it up
so `pytest` works without needing DYLD_LIBRARY_PATH in the shell.
"""

import os
import sys


def pytest_configure(config):
    """Add Homebrew lib path so pyvips can find libvips on macOS."""
    if sys.platform == "darwin":
        homebrew_lib = "/opt/homebrew/lib"
        if os.path.isdir(homebrew_lib):
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            if homebrew_lib not in current:
                os.environ["DYLD_LIBRARY_PATH"] = (
                    f"{homebrew_lib}:{current}" if current else homebrew_lib
                )
