"""Strict, deterministic configuration loading for RoboDojo rollouts."""

from __future__ import annotations

import json
from pathlib import Path

from roboicl.paths import CODE, ROBODOJO


def read_json(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def task_names() -> tuple[str, ...]:
    return tuple(path.stem for path in sorted((CODE / "configs/tasks").glob("*.json")))


def supported_task(name: object) -> bool:
    return isinstance(name, str) and name in task_names()


def load_task(name: str) -> dict:
    path = CODE / "configs/tasks" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown configured RoboDojo task: {name}")
    value = read_json(path)
    return value


def _validate_task_horizons(horizons: object, *, source: Path | str) -> dict[str, int]:
    """Validate one complete adaptive-horizon map and return a copy.

    The map is shared by shot protocols.  Keeping validation here means a new
    protocol can override only the task entries it intentionally changes while
    still failing early when a task is missing or an invalid horizon is added.
    """
    if (not isinstance(horizons, dict)
            or set(horizons) != set(task_names())
            or any(type(value) is not int or not 1 <= value <= 64
                   for value in horizons.values())):
        raise ValueError(
            f"adaptive horizon map must cover every configured task with values in [1,64]: {source}"
        )
    return {str(task): int(value) for task, value in horizons.items()}


def load_adaptive_horizons(path: Path | str | None = None) -> dict[str, int]:
    """Load the repository-wide adaptive horizon map.

    Protocol files share ``configs/adaptive_horizons.json`` by default.  A
    profile may point at another complete map when reproducing a separately
    versioned experiment; profile-level overrides are applied by
    :func:`load_profile` after this function returns.
    """
    source = Path(path) if path is not None else CODE / "configs/adaptive_horizons.json"
    value = read_json(source)
    horizons = value.get("task_horizons") if "task_horizons" in value else value
    return _validate_task_horizons(horizons, source=source)


def load_profile(path: Path | str) -> dict:
    profile_path = Path(path)
    profile = read_json(profile_path)
    if not isinstance(profile.get("id"), str) or not profile["id"].strip():
        raise ValueError("Protocol profile requires a non-empty id")
    harness_overrides = profile.get("harness_overrides", {})
    if not isinstance(harness_overrides, dict):
        raise ValueError("Protocol harness_overrides must be an object")
    shot_count = profile.get("shot_count")
    if shot_count not in (0, 1, None):
        raise ValueError("Published protocols support only shot_count 0 or 1")
    chunk_count = profile.get("train_chunk_count", 0)
    if type(chunk_count) is not int or chunk_count < 0:
        raise ValueError("train_chunk_count must be a nonnegative integer")
    if shot_count == 0 and chunk_count != 0:
        raise ValueError("0-shot protocols cannot contain TRAIN chunks")
    if shot_count == 1 and chunk_count < 1:
        raise ValueError("1-shot protocols must declare train_chunk_count")
    legacy_horizons = profile.get("task_horizons")
    horizon_file = profile.get("adaptive_horizon_file")
    if legacy_horizons is not None and horizon_file is not None:
        raise ValueError("Specify only one of task_horizons and adaptive_horizon_file")
    if legacy_horizons is not None:
        # Accept the pre-composition profile shape while new profiles use the
        # repository-wide map plus optional sparse overrides.
        horizons = _validate_task_horizons(legacy_horizons, source=profile_path)
    elif horizon_file is None:
        horizons = load_adaptive_horizons()
    else:
        if not isinstance(horizon_file, str) or not horizon_file.strip():
            raise ValueError("adaptive_horizon_file must be a non-empty path")
        horizon_path = Path(horizon_file)
        if not horizon_path.is_absolute():
            horizon_path = profile_path.parent / horizon_path
        horizons = load_adaptive_horizons(horizon_path)
    horizon_overrides = profile.get("adaptive_horizon_overrides", {})
    if not isinstance(horizon_overrides, dict):
        raise ValueError("adaptive_horizon_overrides must be an object")
    unknown = set(horizon_overrides) - set(task_names())
    if unknown:
        raise ValueError(f"Unknown adaptive horizon override task(s): {sorted(unknown)}")
    if any(type(value) is not int or not 1 <= value <= 64 for value in horizon_overrides.values()):
        raise ValueError("adaptive horizon overrides must be integers in [1,64]")
    horizons.update(horizon_overrides)
    # Keep the existing in-memory profile contract for callers such as run.py,
    # while removing the duplicated 42-entry map from every protocol JSON.
    profile["task_horizons"] = horizons
    anchors = harness_overrides.get("live_anchor_count")
    if type(anchors) is not int or anchors < 1:
        raise ValueError("Protocol must declare a positive live_anchor_count")
    return profile


def task_source(task: str, variant: str) -> Path:
    if variant not in ("standard", "random"):
        raise ValueError("variant must be standard or random")
    actual = task + ("_random" if variant == "random" else "")
    source = ROBODOJO / "task/RoboDojo/tasks" / f"{actual}.py"
    if not source.is_file():
        raise FileNotFoundError(f"RoboDojo has no {variant} implementation for {task}: {source}")
    return source


def task_name(task: str, variant: str) -> str:
    """Return the exact upstream task module name for a configured task."""
    task_source(task, variant)
    return task + ("_random" if variant == "random" else "")


def layout_path(data_root: Path, task: str, seed: int, layout: int,
                variant: str = "standard") -> Path:
    """Resolve one exact public layout without scanning or renumbering files."""
    if type(seed) is not int or seed < 0 or type(layout) is not int or layout < 0:
        raise ValueError("seed and layout must be nonnegative integers")
    actual = task_name(task, variant)
    return (Path(data_root) / "Assets/Eval_Layout/RoboDojo/arx_x5"
            / str(seed) / f"{actual}_{layout}.json")


def reference_path(data_root: Path, task: str, horizon: int,
                   chunks: int = 12) -> Path:
    """Canonical location of a generated one-shot reference package."""
    if type(horizon) is not int or not 1 <= horizon <= 64:
        raise ValueError("horizon must be in [1,64]")
    if type(chunks) is not int or chunks < 1:
        raise ValueError("chunks must be positive")
    return (Path(data_root) / "runtime-data" / task
            / f"reference_1shot_train{chunks}_endpoint_h{horizon}")


def render_official_task_prompt(task_config: dict) -> tuple[str, str, str]:
    document = task_config.get("official_task_documentation")
    if not isinstance(document, dict):
        raise ValueError("Task config must include official_task_documentation")
    for field in ("version", "source", "description"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"official_task_documentation.{field} must be non-empty")
    scoring = document.get("scoring")
    if not isinstance(scoring, list) or not scoring:
        raise ValueError("official_task_documentation.scoring must be non-empty")
    score_lines = []
    for row in scoring:
        if not isinstance(row, dict) or set(row) != {"score", "condition"}:
            raise ValueError("Each scoring row must contain score and condition")
        score_lines.append(f"- Score {row['score']}: {row['condition']}")
    sections = [f"OFFICIAL DESCRIPTION:\n{document['description'].strip()}"]
    if document.get("comment"):
        sections.append(f"OFFICIAL COMMENT:\n{document['comment'].strip()}")
    sections.append("OFFICIAL SCORING:\n" + "\n".join(score_lines))
    return document["version"], document["source"], "\n\n".join(sections)
