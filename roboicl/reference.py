"""Build or verify a deterministic one-shot, twelve-chunk TRAIN reference."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from roboicl.config import load_task
from roboicl.paths import configure_imports

configure_imports()

import h5py
import numpy as np
from XPolicyLab.utils.process_data import decode_image_bit

from roboicl.policy.astra_policy import CAMERAS, validate_actions
from roboicl.policy.observation_images import encode_camera_rgb, image_encoding
from roboicl.policy.reference_data import state_at, tensor_at
from roboicl.policy.train_reference_bundle import (
    BUNDLE_VARIANT,
    CHILD_VARIANT,
    bundle_digest,
    portable_bundle_digest,
    verify_reference_bundle,
)
from roboicl.policy.train_reference_checks import train_calibration_profile, training_summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_bundle_sha256(bundle: dict) -> str:
    """Hash prompt content while replacing the source locator by its file hash."""
    return portable_bundle_digest(bundle)


def _valid_chunk(data, start: int, horizon: int, translation: float, rotation: float) -> bool:
    rows = np.asarray(tensor_at(data, start, horizon), dtype=float)
    return all(
        np.linalg.norm(rows[:, offset : offset + 3], axis=1).max() <= translation + 1e-6
        and np.linalg.norm(rows[:, offset + 3 : offset + 6], axis=1).max() <= rotation + 1e-6
        for offset in (0, 7)
    )


def select_starts(data, horizon: int, count: int = 12, *,
                  max_translation_m: float = 0.03,
                  max_rotation_rad: float = 0.15) -> list[int]:
    """Select frame zero, the true endpoint chunk, and deterministic interior chunks.

    Interior chunks minimize distance to evenly spaced time targets. All selected
    intervals are non-overlapping and satisfy the LIVE action bounds. Ties are
    resolved by the lexicographically earliest sequence.
    """
    total = len(data["action/left_ee_poses"])
    if type(horizon) is not int or not 1 <= horizon <= 64:
        raise ValueError("horizon must be in [1,64]")
    if type(count) is not int or count < 2:
        raise ValueError("count must be at least two")
    final_start = total - 1 - horizon
    if final_start < (count - 1) * horizon:
        raise ValueError(f"Trajectory has {total} frames; cannot fit {count} disjoint horizon-{horizon} chunks")

    valid = [
        start for start in range(final_start + 1)
        if _valid_chunk(data, start, horizon, max_translation_m, max_rotation_rad)
    ]
    if 0 not in valid or final_start not in valid:
        raise ValueError("Frame-zero or endpoint TRAIN chunk violates the LIVE action bounds")

    candidates = [start for start in valid if horizon <= start <= final_start - horizon]
    needed = count - 2
    targets = [final_start * index / (count - 1) for index in range(1, count - 1)]

    # Dynamic programming avoids greedy dead ends on short or irregular traces.
    states: dict[int, tuple[float, tuple[int, ...]]] = {}
    for slot, target in enumerate(targets):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for start in candidates:
            if slot == 0:
                previous = (0.0, ()) if start >= horizon else None
            else:
                eligible = [value for end, value in states.items() if start >= end + horizon]
                previous = min(eligible, default=None)
            if previous is None:
                continue
            remaining = needed - slot - 1
            if start + horizon * (remaining + 1) > final_start:
                continue
            score = previous[0] + abs(start - target)
            path = previous[1] + (start,)
            incumbent = next_states.get(start)
            if incumbent is None or (score, path) < incumbent:
                next_states[start] = (score, path)
        states = next_states
        if not states:
            raise ValueError("No deterministic non-overlapping J-chunk selection satisfies the action bounds")
    path = min(states.values(), key=lambda value: (value[0], value[1]))[1]
    return [0, *path, final_start]


def _observation(data, frame: int) -> dict:
    images = {
        camera: encode_camera_rgb(decode_image_bit(data[f"vision/{camera}/colors"][frame]), "native_jpeg")
        for camera in CAMERAS
    }
    return {
        "frame": frame,
        "state": state_at(data, frame),
        "head_image": images["cam_head"],
        "prompt_images": images,
    }


def build(task: str, source: Path, output: Path, source_sha256: str,
          horizon: int, chunks: int = 12, source_id: str | None = None) -> dict:
    load_task(task)
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Reference output must be new: {output}")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != source_sha256:
        raise ValueError(f"Source HDF5 SHA256 mismatch: expected {source_sha256}, got {actual_sha256}")
    source_id = source_id or f"runtime-data/{task}/train/{source.name}"
    source_id_path = Path(source_id)
    if source_id_path.is_absolute() or ".." in source_id_path.parts:
        raise ValueError("source_id must be a portable path relative to the data root")
    source_id = source_id_path.as_posix()

    with h5py.File(source, "r") as data:
        frequency = float(data["additional_info/frequency"][()])
        if frequency != 25:
            raise ValueError(f"TRAIN source frequency is {frequency}, expected 25 Hz")
        instruction = data["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        shared = json.loads((Path(__file__).resolve().parents[1] / "configs/shared_harness.json").read_text())
        starts = select_starts(
            data,
            horizon,
            chunks,
            max_translation_m=shared["max_translation_m"],
            max_rotation_rad=load_task(task).get("harness_overrides", {}).get(
                "max_rotation_rad", shared["max_rotation_rad"]
            ),
        )
        examples = []
        for start in starts:
            tensor = tensor_at(data, start, horizon)
            state = state_at(data, start)
            raw = [
                {"mode": "ee_delta", "left": row[:6], "left_gripper": row[6],
                 "right": row[7:13], "right_gripper": row[13]}
                for row in tensor
            ]
            validate_actions(raw, horizon, state)
            example = _observation(data, start)
            result = _observation(data, start + horizon)
            example.update(
                action_tensor=tensor,
                action_step_interval_inclusive=[start, start + horizon - 1],
                result_frame=start + horizon,
                result_state=result["state"],
                result_prompt_images=result["prompt_images"],
            )
            examples.append(example)
        total = len(data["action/left_ee_poses"])
        terminal = _observation(data, total - 1)

    common = {
        "task": task,
        "instruction": instruction,
        "action_space": "ee_delta",
        "delta_frame": "world",
        "frequency": 25,
        "prompt_image_encoding": image_encoding("native_jpeg", "original", "triptych"),
    }
    child = {
        **deepcopy(common),
        "variant": CHILD_VARIANT,
        "source_file": source_id,
        "sha256": actual_sha256,
        "source_episode": int(source.stem.rsplit("_", 1)[-1]),
        "total_frames": total,
        "examples": examples,
        "terminal_observation": terminal,
        "geometry_profile": train_calibration_profile(),
        "prompt_cameras": list(CAMERAS),
        "demo_message_format": "tool_calls",
        "requires_initial_frame": True,
        "prediction_horizon": horizon,
        "selected_action_starts": starts,
        "selection": (
            "Deterministic J-chunk selection: frame zero, true endpoint chunk, and evenly spaced "
            "non-overlapping interior windows; "
            "all labels and RGB/state observations are re-read from the hash-locked source."
        ),
    }
    bundle = {
        **deepcopy(common),
        "variant": BUNDLE_VARIANT,
        "episodes": [child],
        "examples": deepcopy(examples),
        "demo_message_format": "tool_calls",
        "requires_initial_frame": True,
        "prompt_cameras": list(CAMERAS),
        "geometry_profile": train_calibration_profile(),
        "source_sha256s": [actual_sha256],
        "source_episode_list": [child["source_episode"]],
        "n_shots": 1,
        "prediction_horizon": horizon,
        "train_chunk_count": chunks,
        "train_image_count": chunks * 2,
        "endpoint_alignment": {
            "schema": "roboicl.train_selection.v1",
            "chunks_per_episode": chunks,
            "terminal_observation_is_last_chunk_result": True,
        },
    }
    bundle["sha256"] = bundle_digest(bundle)
    verification_bundle = deepcopy(bundle)
    verification_bundle["episodes"][0]["source_file"] = str(source)
    verification_bundle["sha256"] = bundle_digest(verification_bundle)
    report = verify_reference_bundle(verification_bundle, train_calibration_profile())
    report["source_sha256"] = bundle["sha256"]
    output.mkdir(parents=True)
    values = {
        "train_reference_bundle.json": bundle,
        "verification.json": training_summary(report),
    }
    for name, value in values.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {
        "task": task,
        "source": str(source),
        "source_sha256": actual_sha256,
        "output": str(output),
        "horizon": horizon,
        "train_chunk_count": chunks,
        "selected_action_starts": starts,
        "bundle_sha256": bundle["sha256"],
        "passed": True,
    }


def verify(directory: Path, *, chunks: int = 12) -> dict:
    directory = directory.expanduser().resolve()
    bundle = json.loads((directory / "train_reference_bundle.json").read_text(encoding="utf-8"))
    if len(bundle.get("episodes", [])) != 1:
        raise ValueError("1-shot reference must contain exactly one source episode")
    if any(len(episode.get("examples", [])) != chunks for episode in bundle["episodes"]):
        raise ValueError(f"Every source episode must contain exactly J={chunks} chunks")
    report = verify_reference_bundle(bundle)
    return {"directory": str(directory), "bundle_sha256": bundle["sha256"], "chunks": chunks,
            "passed": report.get("passed") is True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--task", required=True)
    builder.add_argument("--source", required=True, type=Path)
    builder.add_argument("--source-sha256", required=True)
    builder.add_argument("--output", required=True, type=Path)
    builder.add_argument("--horizon", required=True, type=int)
    builder.add_argument("--chunks", type=int, default=12)
    builder.add_argument(
        "--source-id",
        help="Portable source path relative to the data root; defaults to runtime-data/<task>/train/<file>",
    )
    checker = subparsers.add_parser("verify")
    checker.add_argument("--reference", required=True, type=Path)
    checker.add_argument("--chunks", type=int, default=12)
    args = parser.parse_args()
    result = (build(args.task, args.source, args.output, args.source_sha256, args.horizon,
                    args.chunks, args.source_id)
              if args.command == "build" else verify(args.reference, chunks=args.chunks))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
