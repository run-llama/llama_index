"""Download pypi package."""

import importlib
import subprocess
import sys
from typing import Any


def pip_install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def download_integration(module_str: str, module_import_str: str, cls_name: str) -> Any:
    """Returns an integration class by first pip installing its parent module."""
    try:
        pip_install(module_str)  # this works for any integration not just packs
    except Exception as e:
        raise Exception(f"Failed to pip install `{module_str}`") from e

    try:
        module_spec = importlib.util.find_spec(module_import_str)
        module = importlib.util.module_from_spec(module_spec)  # type: ignore
        module_spec.loader.exec_module(module)  # type: ignore
        pack_cls = getattr(module, cls_name)
    except ImportError as e:
        raise ImportError(f"Unable to import {cls_name}") from e
    return pack_cls
