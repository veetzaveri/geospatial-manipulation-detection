from __future__ import annotations

import os
import shutil
from pathlib import Path


def place_file(source: Path, destination: Path, method: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if method == "hardlink":
        os.link(source, destination)
    elif method == "symlink":
        destination.symlink_to(source.resolve())
    elif method == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("method must be hardlink, symlink, or copy")
