"""Verify pinned upstream submodules and RoboICL's integration commits."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from roboicl.paths import CODE, configure_imports


def verify_submodules(lock_path: Path | None = None) -> dict:
    lock_path = lock_path or CODE / "configs/upstream.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checked = []
    for package in lock["packages"]:
        path = CODE / package["path"]
        if not path.is_dir():
            raise FileNotFoundError(f"Missing submodule {path}; run git submodule update --init --recursive")
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != package["commit"]:
            raise ValueError(f"{package['name']} is at {actual}; expected {package['commit']}")
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise ValueError(f"Submodule must remain clean: {path}")
        checked.append({"name": package["name"], "path": package["path"], "commit": actual})
    return {"schema": lock["schema"], "packages": checked}


def activate() -> None:
    configure_imports()
