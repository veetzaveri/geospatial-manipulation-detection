from __future__ import annotations

import os
import platform
import random
import sys
import warnings

import numpy as np
import torch

_NONDETERMINISTIC_WARNINGS: list[str] = []
_WARNING_HOOK_INSTALLED = False


def _install_determinism_warning_recorder() -> None:
    global _WARNING_HOOK_INSTALLED
    if _WARNING_HOOK_INSTALLED:
        return
    previous_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        rendered = str(message)
        lowered = rendered.lower()
        if "nondeterministic" in lowered or "deterministic implementation" in lowered:
            _NONDETERMINISTIC_WARNINGS.append(rendered)
        previous_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
    _WARNING_HOOK_INSTALLED = True


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Apply the seed and deterministic settings specified in Table 2."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        _install_determinism_warning_recorder()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def runtime_metadata() -> dict:
    """Record the actual software/hardware and determinism state for each run."""

    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as error:  # pragma: no cover - environment-specific
        torchvision_version = f"unavailable: {error}"

    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    warn_only_check = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": gpu_names,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": warn_only_check() if warn_only_check else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nondeterministic_operation_warnings": list(
            dict.fromkeys(_NONDETERMINISTIC_WARNINGS)
        ),
    }


def dataloader_generator(seed: int = 42) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
