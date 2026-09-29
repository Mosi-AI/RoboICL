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
from roboicl.reference import build, portable_bundle_sha256, verify


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

    lock = json.loads((CODE / "configs/reference_sources.lock.json").read_text(encoding="utf-8"))
    matches = [row for row in lock["references"] if row["task"] == args.task]
    if len(matches) != 1:
        raise ValueError(
            f"No unique published J=12 reference for {args.task!r}; "
            "use `python -m roboicl.reference build` with an explicit source HDF5"
        )
    row = matches[0]
    repository = lock["repository"]
    # Some published references pin the source episode to a different commit
    # than the repository-level protocol metadata.  The episode pin is
    # authoritative; falling back to the repository revision preserves the
    # original behavior for older lock records.
    source_revision = row["source"].get("repository_revision", repository["revision"])
    data_root = args.data_root.expanduser().resolve()
    source = data_root / row["source"]["install_path"]
    output = data_root / "runtime-data" / row["path"]
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
        if partial.stat().st_size != row["source"]["size"]:
            raise ValueError(f"Downloaded source size mismatch: {partial}")
        if file_sha256(partial) != row["source"]["sha256"]:
            raise ValueError(f"Downloaded source SHA256 mismatch: {partial}")
        partial.replace(source)

    if output.exists():
        report = verify(output, chunks=row["chunks"])
    else:
        report = build(
            args.task,
            source,
            output,
            row["source"]["sha256"],
            row["horizon"],
            row["chunks"],
            row["source"]["install_path"],
        )
    bundle = json.loads((output / "train_reference_bundle.json").read_text(encoding="utf-8"))
    portable = portable_bundle_sha256(bundle)
    if row.get("portable_sha256") and portable != row["portable_sha256"]:
        raise ValueError(f"Generated reference differs from the published lock: {output}")
    report.update(reference=str(output), portable_sha256=portable)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
