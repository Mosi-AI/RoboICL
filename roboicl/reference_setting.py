"""Load and validate a paper reference-setting manifest."""

from __future__ import annotations

import json
from pathlib import Path

from roboicl.paths import CODE


def load_reference_setting(path: str | Path) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = CODE / manifest_path
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema") != "roboicl.reference_setting.v1":
        raise ValueError(f"Unsupported reference-setting schema: {manifest_path}")
    zero = value.get("zero_shot_tasks")
    refs = value.get("one_shot_references")
    if not isinstance(zero, list) or not isinstance(refs, list):
        raise ValueError("Reference setting must declare zero_shot_tasks and one_shot_references")
    zero_names = [row["task"] for row in zero]
    ref_names = [row["task"] for row in refs]
    if len(zero_names) != 8 or len(ref_names) != 22 or len(set(zero_names + ref_names)) != 30:
        raise ValueError("Paper setting must contain 8 unique zero-shot and 22 unique one-shot tasks")
    chunks = value.get("train", {}).get("blocks_per_demonstration")
    anchors = value.get("live", {}).get("retained_blocks")
    if chunks != 12 or anchors != 12:
        raise ValueError("This setting must declare TRAIN J=12 and LIVE B=12")
    for row in refs:
        if any(key in row for key in ("evaluation_status", "exception")):
            raise ValueError(f"Public reference rows cannot contain status annotations: {row.get('task')}")
        source = row.get("source", {})
        if "bundle_source_episode" in source:
            raise ValueError(f"Public reference rows cannot contain bundle-only episode aliases: {row.get('task')}")
        for key in ("install_path", "repository_path"):
            field_value = source.get(key)
            field_path = Path(field_value) if isinstance(field_value, str) else Path()
            if (not isinstance(field_value, str) or not field_value or field_path.is_absolute()
                    or ".." in field_path.parts):
                raise ValueError(f"Non-portable {key} for {row.get('task')}")
        starts = row.get("selected_action_starts")
        horizon = row.get("reference_horizon")
        total = source.get("total_frames")
        episode = source.get("episode")
        install_name = Path(source.get("install_path", "")).stem
        if (type(episode) is not int or not install_name.startswith("episode_")
                or install_name.rsplit("_", 1)[-1] != f"{episode:07d}"):
            raise ValueError(f"Installed source episode does not match its path for {row.get('task')}")
        if (not isinstance(starts, list) or len(starts) != chunks
                or any(type(item) is not int for item in starts)
                or starts != sorted(set(starts))):
            raise ValueError(f"Invalid locked TRAIN starts for {row.get('task')}")
        if type(horizon) is not int or type(total) is not int:
            raise ValueError(f"Invalid horizon/source length for {row.get('task')}")
        if any(right - left < horizon for left, right in zip(starts, starts[1:])):
            raise ValueError(f"Overlapping TRAIN windows for {row['task']}")
        if starts[-1] + horizon >= total:
            raise ValueError(f"TRAIN result frame is outside the source for {row['task']}")
    value["_path"] = str(manifest_path.resolve())
    return value


def reference_record(setting: dict, task: str) -> dict:
    matches = [row for row in setting["one_shot_references"] if row["task"] == task]
    if len(matches) != 1:
        raise ValueError(f"No unique one-shot reference in {setting['id']} for {task!r}")
    return matches[0]
