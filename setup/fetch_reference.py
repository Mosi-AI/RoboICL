#!/usr/bin/env python3
"""Fetch one locked RoboDojo source episode and build its RoboICL reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from roboicl.config import reference_path
from roboicl.reference import build, verify
from roboicl.reference_setting import load_reference_setting, reference_record


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--data-root", type=Path, default=CODE / "data")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    args = parser.parse_args()

    setting = load_reference_setting("configs/references/one_shot_j12_b12.json")
    row = reference_record(setting, args.task)
    repository = setting["source_repository"]
    source_revision = row["source"].get("repository_revision", repository["revision"])
    data_root = args.data_root.expanduser().resolve()
    source = data_root / row["source"]["install_path"]
    output = reference_path(data_root, args.task, row["reference_horizon"], 12)
    os.environ["ROBOICL_DATA_ROOT"] = str(data_root)

    source.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        actual = file_sha256(source)
        if actual != row["source"]["sha256"]:
            raise ValueError(f"Existing source preserved but its SHA256 is wrong: {source}")
    else:
        partial = source.with_suffix(source.suffix + ".partial")
        if partial.exists():
            raise RuntimeError(f"Partial download preserved for inspection: {partial}")
        url = (
            args.endpoint.rstrip("/")
            + f"/datasets/{repository['repo_id']}/resolve/{source_revision}/"
            + row["source"]["repository_path"]
        )
        urllib.request.urlretrieve(url, partial)
        if row["source"].get("size") is not None and partial.stat().st_size != row["source"]["size"]:
            raise ValueError(f"Downloaded source size mismatch: {partial}")
        if file_sha256(partial) != row["source"]["sha256"]:
            raise ValueError(f"Downloaded source SHA256 mismatch: {partial}")
        partial.replace(source)

    if output.exists():
        report = verify(output, chunks=setting["train"]["blocks_per_demonstration"])
        bundle = json.loads((output / "train_reference_bundle.json").read_text(encoding="utf-8"))
        episode = bundle["episodes"][0]
        if (episode.get("sha256") != row["source"]["sha256"]
                or episode.get("selected_action_starts") != row["selected_action_starts"]
                or bundle.get("prediction_horizon") != row["reference_horizon"]):
            raise ValueError(f"Existing reference does not match {setting['id']}: {output}")
    else:
        report = build(
            args.task,
            source,
            output,
            row["source"]["sha256"],
            row["reference_horizon"],
            setting["train"]["blocks_per_demonstration"],
            row["source"]["install_path"],
            row["selected_action_starts"],
            setting["id"],
        )
    report.update(reference=str(output), reference_setting_id=setting["id"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
