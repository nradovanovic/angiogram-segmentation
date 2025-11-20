from __future__ import annotations

import os
from pathlib import Path


def resolve_data_dir(data_dir: str | Path) -> Path:
    """
    Resolve the dataset directory, with Kaggle auto-discovery fallback.

    Kaggle datasets are mounted read-only under /kaggle/input/<slug>/.
    When running inside that environment we search for a folder that matches
    the requested data directory name so users don't have to hard-code paths.
    """
    path = Path(data_dir)
    if path.exists():
        return path

    kaggle_root = Path("/kaggle/input")
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ and kaggle_root.exists():
        matches = sorted(kaggle_root.glob(f"**/{path.name}"))
        if matches:
            print(f"[INFO] Resolved data directory to {matches[0]} (Kaggle input).")
            return matches[0]

    raise FileNotFoundError(
        f"Could not locate data directory '{data_dir}'. "
        "Pass --data_dir with an existing path or attach the dataset in Kaggle."
    )

