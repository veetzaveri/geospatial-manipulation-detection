from __future__ import annotations

import hashlib
from pathlib import Path


def stable_seed(value: str | Path, global_seed: int = 42) -> int:
    digest = hashlib.sha256(f"{global_seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)
