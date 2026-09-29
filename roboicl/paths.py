"""Repository, data, result, and pinned upstream import paths."""

from __future__ import annotations

import os
from pathlib import Path
import sys


CODE = Path(__file__).resolve().parents[1]
THIRD_PARTY = CODE / "third_party"
ROBODOJO = THIRD_PARTY / "RoboDojo"
XPOLICYLAB = THIRD_PARTY / "XPolicyLab"
ISAACLAB = THIRD_PARTY / "IsaacLab"
CUROBO = THIRD_PARTY / "curobo"


def data_root(value=None) -> Path:
    return Path(value or os.environ.get("ROBOICL_DATA_ROOT") or CODE / "data").expanduser().resolve()


def results_root(value=None) -> Path:
    return Path(value or os.environ.get("ROBOICL_RESULTS_ROOT") or CODE / "results").expanduser().resolve()


def python_paths(context: str = "policy") -> list[str]:
    """Paths needed to import owned code and the four clean submodules.

    RoboDojo and XPolicyLab both expose a historical top-level ``utils``
    package. Put the process owner first so Python resolves that namespace
    without copying or modifying either submodule.
    """
    if context not in ("policy", "sim"):
        raise ValueError("context must be policy or sim")
    owners = [XPOLICYLAB, ROBODOJO] if context == "policy" else [ROBODOJO, XPOLICYLAB]
    paths = [CODE, THIRD_PARTY, *owners, CUROBO]
    paths.extend(path for path in sorted((ISAACLAB / "source").glob("*")) if path.is_dir())
    return [str(path) for path in paths]


def configure_imports(context: str = "policy") -> None:
    paths = python_paths(context)
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
