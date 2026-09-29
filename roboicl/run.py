"""One shared launcher for RoboICL tasks; no API request is made by --dry-run."""

import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


CODE = Path(__file__).absolute().parents[1]
GPU_SLOTS = 2
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from roboicl.config import (
    layout_path as configured_layout_path,
    load_profile,
    load_task,
    reference_path as configured_reference_path,
    render_official_task_prompt,
    task_name,
    task_source,
)
from roboicl.policy.image_budget import (
    anchored_image_budget, reference_image_count,
)
from roboicl.policy.provider_compat import validate_api_mode, validate_endpoint
from roboicl.paths import data_root as get_data_root, results_root as get_results_root, python_paths
from roboicl.policy.train_reference_bundle import portable_bundle_digest
from roboicl.upstream import verify_submodules


def absolute_path(value):
    return Path(value).expanduser().absolute()


def python_path(value):
    """Accept an interpreter path or a command discoverable on PATH."""
    found = shutil.which(str(value))
    path = absolute_path(found or value)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {path}")
    return path


def policy_socket_ready(port):
    """Perform a WebSocket handshake without installing launch-time packages."""
    with socket.create_connection(("127.0.0.1", port), timeout=1) as probe:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        probe.sendall((f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                       "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                       f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = probe.recv(4096)
            if not chunk or len(response) > 8192:
                raise OSError("Policy did not return a WebSocket handshake")
            response += chunk
        if not response.startswith(b"HTTP/1.1 101 "):
            raise OSError("Policy did not accept the WebSocket handshake")
        # Empty masked close frame; no application request is ever sent.
        probe.sendall(b"\x88\x80" + os.urandom(4))


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_preflight(data_root, layout_path):
    lock = json.loads((CODE / "configs/data.lock.json").read_text())["assets"]
    manifest_path = data_root / ".roboicl-assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run setup/fetch_assets.py"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != lock["repo_id"] or manifest.get("revision") != lock["revision"]:
        raise ValueError(f"Asset dataset revision mismatch: {manifest_path}")
    scope = manifest.get("scope", "complete-tree")
    if scope == "explicit-files":
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"Explicit asset manifest contains no files: {manifest_path}")
        checked = []
        for row in files:
            relative = Path(row.get("path", ""))
            if (not relative.parts or relative.parts[0] != "Assets" or relative.is_absolute()
                    or ".." in relative.parts or type(row.get("size")) is not int
                    or not isinstance(row.get("sha256"), str)):
                raise ValueError(f"Invalid explicit asset record: {row}")
            path = data_root / relative
            if not path.is_file() or path.stat().st_size != row["size"] or sha(path) != row["sha256"]:
                raise ValueError(f"Explicit asset differs from its manifest: {path}")
            checked.append(relative.as_posix())
        selected = layout_path.relative_to(data_root).as_posix()
        if selected not in checked:
            raise ValueError(f"Selected layout is outside the explicit asset manifest: {layout_path}")
    elif scope != "complete-tree" or manifest.get("include") != lock["include"]:
        raise ValueError(f"Unknown or incomplete asset manifest scope: {manifest_path}")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
        "repo_id": lock["repo_id"],
        "revision": lock["revision"],
        "scope": scope,
        "assets_tree_sha256": manifest.get("assets_tree_sha256"),
    }


def reference_preflight(task, shots, config, ref, profile, data_root):
    """Check immutable reference provenance without simulator/HDF5 imports.

    Preparation performs source RGB/state verification; here we verify its
    content-addressed bundle/report and re-hash the source files before launch.
    The policy still performs its full manifest validation when it starts.
    """
    bundle = None
    report = {"shots": shots, "source_files": []}
    if shots:
        bundle = json.loads((ref / "train_reference_bundle.json").read_text())
        canonical = {key: value for key, value in bundle.items() if key != "sha256"}
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if bundle.get("sha256") != digest:
            raise ValueError(f"TRAIN bundle content hash mismatch: {ref}")
        if bundle.get("task") != task:
            raise ValueError(f"TRAIN task {bundle.get('task')!r} does not match {task!r}")
        episodes = bundle.get("episodes", [])
        if len(episodes) != shots:
            raise ValueError(f"Requested {shots}-shot but TRAIN bundle contains {len(episodes)} episodes: {ref}")
        expected_chunks = profile.get("train_chunk_count")
        if (type(expected_chunks) is not int or expected_chunks < 1
                or any(len(episode.get("examples", [])) != expected_chunks for episode in episodes)):
            raise ValueError(f"TRAIN reference must contain exactly J={expected_chunks} chunks per episode: {ref}")
        if bundle.get("train_chunk_count") != expected_chunks:
            raise ValueError(f"TRAIN bundle does not declare train_chunk_count={expected_chunks}: {ref}")
        if bundle.get("prediction_horizon") != config["predict_horizon"]:
            raise ValueError(f"TRAIN bundle horizon does not match the selected protocol: {ref}")
        verification = json.loads((ref / "verification.json").read_text())
        if verification.get("passed") is not True or verification.get("source_sha256") != digest:
            raise ValueError(f"TRAIN verification is stale or failed: {ref}")
        checked_sources = set()
        for episode in episodes:
            examples = episode.get("examples", [])
            if not examples or any(len(example.get("action_tensor", [])) != config["predict_horizon"]
                                   for example in examples):
                raise ValueError(f"TRAIN action chunks do not match requested horizon: {ref}")
            for example in examples:
                validate_train_actions(example["action_tensor"], config)
                if config.get("train_result_observations") and set(
                    example.get("result_prompt_images", {})
                ) != {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
                    raise ValueError(f"TRAIN chunk is missing embedded result camera images: {ref}")
            source = Path(episode["source_file"]).expanduser()
            if not source.is_absolute():
                source = data_root / source
            elif not source.is_file():
                # Historical bundles can retain an absolute path from the
                # source server. Prefer the canonical local task location;
                # the locked episode SHA256 below remains authoritative.
                candidate = (data_root / "runtime-data" / task / "train"
                             / source.name)
                if candidate.is_file():
                    source = candidate
            if episode.get("sha256") in checked_sources:
                raise ValueError(f"TRAIN bundle repeats a source episode: {source}")
            checked_sources.add(episode.get("sha256"))
            if source.is_file():
                actual = sha(source)
                if actual != episode.get("sha256"):
                    raise ValueError(f"TRAIN source SHA256 mismatch: {source}")
                report["source_files"].append({"path": str(source), "sha256": actual,
                                                "available": True})
            else:
                # A portable, independently checked bundle may be replayed
                # without shipping the original HDF5.  Require the persisted
                # verification sidecar to bind the episode source hash and
                # successful source audit; if the source later appears, the
                # branch above re-hashes it and remains authoritative.
                episode_reports = verification.get("episodes", [])
                matching = [item for item in episode_reports
                            if item.get("source_sha256") == episode.get("sha256")]
                if len(matching) != 1 or matching[0].get("passed") is not True:
                    raise FileNotFoundError(
                        f"TRAIN source is absent and verification sidecar has no passed audit: {source}"
                    )
                report["source_files"].append({"path": str(source),
                                                "sha256": episode.get("sha256"),
                                                "available": False,
                                                "status": "offline_verified_bundle"})
        report.update(bundle_sha256=digest, verification_sha256=sha(ref / "verification.json"))
        lock = json.loads((CODE / "configs/reference_sources.lock.json").read_text())
        try:
            relative = ref.relative_to(data_root).as_posix()
        except ValueError:
            relative = None
        if relative and relative.startswith("runtime-data/"):
            relative = relative.removeprefix("runtime-data/")
        records = [row for row in lock.get("references", [])
                   if row.get("task") == task and row.get("shots") == shots
                   and row.get("path") == relative]
        if records:
            locked = records[0]
            if locked.get("horizon") != config["predict_horizon"]:
                raise ValueError(f"Reference lock horizon mismatch for {task}/{shots}-shot")
            if locked.get("portable_sha256") != portable_bundle_digest(bundle):
                raise ValueError(f"Reference payload differs from configs/reference_sources.lock.json: {ref}")
            report["reference_lock"] = {
                "path": relative or str(ref),
                "portable_sha256": locked["portable_sha256"],
            }
        else:
            report["reference_lock"] = {
                "path": relative,
                "portable_sha256": portable_bundle_digest(bundle),
                "status": "local hash-locked reference not yet listed in the published lock",
            }
    cameras = config.get("demo_cameras", ["cam_head", "cam_left_wrist", "cam_right_wrist"])
    layout = config.get("image_layout", "triptych")
    fixed = reference_image_count(bundle, cameras=cameras,
                                  frame0_cameras=config.get("demo_frame0_cameras", ()),
                                  image_layout=layout,
                                  include_result_observations=config.get("train_result_observations", False))
    if config.get("live_anchor_count", 0):
        report["image_budget"] = anchored_image_budget(
            fixed, config["live_anchor_count"], config.get("feedback_cameras", cameras), layout,
            retain_latest=config.get("live_retain_latest", False))
        cap = config.get("max_context_images")
        if cap is not None and report["image_budget"]["max_total_images"] > cap:
            raise ValueError(f"TRAIN + LIVE worst-case image budget exceeds {cap}: {report['image_budget']}")
    else:
        report["image_budget"] = {"train_images": fixed, "max_total_images": None,
                                  "note": "No static bound for non-anchored history; policy enforces request limits."}
    return report


def validate_train_actions(rows, config):
    """Apply the LIVE numeric contract to every immutable TRAIN action row."""
    translation = config.get("max_translation_m", .03)
    rotation = config.get("max_rotation_rad", .15)
    if any(type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0
           for limit in (translation, rotation)):
        raise ValueError("TRAIN/LIVE action norm limits must be positive finite numbers")
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 14:
            raise ValueError(f"TRAIN action row {index} must contain 14 numbers")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
            raise ValueError(f"TRAIN action row {index} must contain finite numbers")
        for offset in (0, 7):
            if not 0 <= row[offset + 6] <= 1:
                raise ValueError(f"TRAIN action row {index} has gripper outside [0,1]")
            if (math.sqrt(sum(value * value for value in row[offset:offset + 3])) > translation + 1e-6
                    or math.sqrt(sum(value * value for value in row[offset + 3:offset + 6])) > rotation + 1e-6):
                raise ValueError(f"TRAIN action row {index} exceeds the LIVE action norm limits")


def check(task, shots, seed, layout, variant, action_horizon=None, *,
          data_root=None, sim_python=None, policy_python=None, profile=None,
          reference=None, preflight_report=None):
    data_root = get_data_root(data_root)
    sim_python = python_path(sim_python or os.environ.get("ROBOICL_SIM_PYTHON", sys.executable))
    policy_python = python_path(policy_python or os.environ.get("ROBOICL_POLICY_PYTHON", sys.executable))
    if profile is None:
        profile = load_profile(os.environ.get("ROBOICL_PROFILE", CODE / "configs/protocols/zero_shot_b25.json"))
    validate_api_mode(profile.get("api_mode", "responses"))
    if "endpoint" in profile:
        validate_endpoint(profile.get("api_mode", "responses"), profile["endpoint"])
    c = load_task(task)
    if profile.get("shot_count") is not None and shots != profile["shot_count"]:
        raise ValueError(f"Protocol {profile['id']} requires shots={profile['shot_count']}")
    task_horizons = profile.get("task_horizons")
    if task not in task_horizons:
        raise ValueError(f"Task-adaptive horizon profile has no entry for {task}")

    actual = task_name(task, variant)

    paths = [
        task_source(task, variant),
        configured_layout_path(data_root, task, seed, layout, variant),
        sim_python,
        policy_python,
    ]
    config = json.loads((CODE / "configs/shared_harness.json").read_text())
    config.update(profile.get("harness_overrides", {}))
    # Task contracts refine the shared/protocol defaults (for example action
    # bounds or a legacy reference's optional result-camera payload).  Keeping
    # this last makes those task-specific constraints authoritative without
    # duplicating the common harness in every task file.
    config.update(c.get("harness_overrides", {}))
    config.update(predict_horizon=task_horizons[task], execute_horizon=task_horizons[task])
    document_version, document_source, document_prompt = render_official_task_prompt(c)
    config["official_task_documentation_version"] = document_version
    config["official_task_documentation_source"] = document_source
    config["official_task_prompt"] = document_prompt
    aliases = c.get("reference_instruction_aliases", {}).get(variant, [])
    if (not isinstance(aliases, list)
            or any(not isinstance(value, str) or not value for value in aliases)
            or len(set(aliases)) != len(aliases)):
        raise ValueError("reference_instruction_aliases must map variants to unique nonempty strings")
    config["reference_instruction_aliases"] = aliases
    if action_horizon is not None:
        config.update(predict_horizon=action_horizon, execute_horizon=action_horizon)
    horizon = config["predict_horizon"]
    if not 1 <= horizon <= 64:
        raise ValueError("action horizon must be 1..64")
    ref = (
        absolute_path(reference)
        if reference is not None
        else configured_reference_path(data_root, task, horizon, profile["train_chunk_count"])
        if shots
        else data_root
    )
    if shots:
        paths += [ref / 'train_reference_bundle.json', ref / 'verification.json']
    for p in paths:
        if not p.exists():
            if shots and p.parent == ref:
                raise FileNotFoundError(f"{p}; build/import a verified J={profile['train_chunk_count']} reference")
            raise FileNotFoundError(p)
    report = reference_preflight(task, shots, config, ref, profile, data_root)
    if preflight_report is not None:
        preflight_report.update(report)
    return c, actual, config, ref, paths[1]


def stop(p):
    if p is None or p.poll() is not None:
        return

    for sig, timeout in [
        (signal.SIGINT, 30),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ]:
        try:
            os.killpg(p.pid, sig)
            p.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def acquire_gpu_slot(gpu, requested_slot=None, *, results_root=None):
    results_root = get_results_root(results_root)
    slots = [requested_slot] if requested_slot is not None else range(GPU_SLOTS)
    # The results directory is shared by multiple machines whose CUDA indices
    # both start at zero. Keep slot locks host-local so GPU 0 on one worker does
    # not block GPU 0 on another worker.
    lock_host = socket.gethostname()
    for slot in slots:
        lock = (results_root / f".{lock_host}.gpu{gpu}.slot{slot}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
        else:
            return slot, lock

    if requested_slot is None:
        raise RuntimeError(f"Both RoboICL slots on GPU {gpu} are occupied")
    raise RuntimeError(f"RoboICL slot {requested_slot} on GPU {gpu} is occupied")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task")
    ap.add_argument("--shots", type=int, choices=[0, 1], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument(
        "--variant", choices=["standard", "random"], default="standard"
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-slot", type=int, choices=range(GPU_SLOTS))
    ap.add_argument("--model", help="CLI override; otherwise ASTRA_MODEL, then profile model")
    ap.add_argument("--harness-profile", "--profile", dest="harness_profile",
                    default=os.environ.get("ROBOICL_PROFILE", str(CODE / "configs/protocols/zero_shot_b25.json")))
    ap.add_argument("--data-root", default=str(get_data_root()),
                    help="Directory containing Assets/ and runtime-data/")
    ap.add_argument("--results-root", default=str(get_results_root()))
    ap.add_argument("--reference", type=Path,
                    help="Explicit verified reference directory; required for tasks without a configured 1-shot bundle")
    ap.add_argument("--sim-python", default=os.environ.get("ROBOICL_SIM_PYTHON", sys.executable))
    ap.add_argument("--policy-python", default=os.environ.get("ROBOICL_POLICY_PYTHON", sys.executable))
    ap.add_argument("--conda-exe", default=os.environ.get("ROBOICL_CONDA_EXE"),
                    help="Optional conda executable for activating each interpreter's environment")
    ap.add_argument("--port", type=int)
    ap.add_argument("--action-horizon", type=int, default=None, help="Override both prediction and execution steps; few-shot needs matching reference")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    output_group = ap.add_mutually_exclusive_group()
    output_group.add_argument("--max-output-tokens", type=int, default=None)
    output_group.add_argument("--omit-max-output-tokens", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--capture-only",
        action="store_true",
        help="Render diagnostic only; no policy API or robot actions",
    )
    args = ap.parse_args()

    if min(args.seed, args.layout, args.gpu) < 0:
        ap.error("seed/layout/gpu must be nonnegative")
    if args.reference is not None and args.shots == 0:
        ap.error("--reference is only valid with --shots 1")

    data_root = absolute_path(args.data_root)
    results_root = absolute_path(args.results_root)
    sim_python = python_path(args.sim_python)
    policy_python = python_path(args.policy_python)
    profile_path = absolute_path(args.harness_profile)
    profile = load_profile(profile_path)
    args.model = args.model or os.environ.get("ASTRA_MODEL") or profile.get("model", "gpt-6-astra")
    api_mode = profile.get("api_mode", "responses")
    if profile.get("launch_blocked_reason") and not (args.dry_run or args.capture_only):
        raise RuntimeError(
            "Provider profile is launch-blocked: " + profile["launch_blocked_reason"])
    conda_exe = python_path(args.conda_exe) if args.conda_exe else None

    preflight_report = {}
    c, task, config, ref, layout_path = check(
        args.task, args.shots, args.seed, args.layout, args.variant, args.action_horizon,
        data_root=data_root, sim_python=sim_python, policy_python=policy_python, profile=profile,
        reference=args.reference,
        preflight_report=preflight_report,
    )
    asset_report = asset_preflight(data_root, layout_path)

    if args.reasoning_effort is not None:
        config["reasoning_effort"] = args.reasoning_effort
    if args.max_output_tokens is not None:
        if not 256 <= args.max_output_tokens <= 65536:
            ap.error("max-output-tokens must be 256..65536")
        config["max_output_tokens"] = args.max_output_tokens
    if args.omit_max_output_tokens:
        config.pop("max_output_tokens", None)
    effort = config.get("reasoning_effort", "xhigh")
    horizon_tag = f"pred{config['predict_horizon']}_exec{config['execute_horizon']}"
    live_anchor_tag = f"livek{config.get('live_anchor_count', 0)}"
    output_tag = "out" + str(config.get("max_output_tokens") or "default")
    task_doc_tag = f"taskdoc-{config.get('official_task_documentation_version', 'none')}"

    with socket.socket() as s:
        s.bind(("127.0.0.1", args.port or 0))
        port = s.getsockname()[1]

    runid = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_{args.task}_{args.shots}shot_s{args.seed}_{args.variant}_l{args.layout}_reasoning-{effort}_{output_tag}_{horizon_tag}_{live_anchor_tag}_{task_doc_tag}"
    )
    out = results_root / runid

    env = os.environ.copy()
    if profile.get("endpoint") is not None:
        # A selected profile is authoritative. In particular, do not inherit a
        # /v1/responses endpoint from a shell credential file for a Chat model.
        env["ASTRA_ENDPOINT"] = profile["endpoint"]
    for key in [
        "ASTRA_DEMO_MANIFEST",
        "ASTRA_CAMERA_PREFLIGHT",
        "ROBODOJO_FATAL_RESTART_COUNT",
        "ASTRA_HARNESS_CONFIG",
        "ASTRA_LOG_DIR",
        "ASTRA_OMIT_MAX_OUTPUT_TOKENS",
    ]:
        env.pop(key, None)

    env.update(
        ROBOICL_DATA_ROOT=str(data_root),
        ROBOICL_RESULTS_ROOT=str(results_root),
        ROBODOJO_RUN_ID=runid,
        ASTRA_HARNESS_CONFIG=str(out / "harness.json"),
        ASTRA_LOG_DIR=str(out / "astra"),
        ASTRA_REASONING_EFFORT=effort,
        ASTRA_API_MODE=api_mode,
        ASTRA_TRACKING_GUARD="1",
        ASTRA_VALIDATE_RGB="1",
        ASTRA_NETWORK_MODE="direct",
        ASTRA_STREAM="1",
        ASTRA_RETRIEVABLE_HISTORY="0",
        PYTHONPATH=os.pathsep.join(python_paths("policy")),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        OMNI_KIT_ALLOW_ROOT="1",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
    )
    if args.shots:
        env["ASTRA_DEMO_MANIFEST"] = str(ref / "train_reference_bundle.json")

    server = [
        str(policy_python),
        "-u",
        "-m",
        "roboicl.policy_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        args.model,
        "--task",
        task,
        "--seed",
        str(args.seed),
        "--reasoning-effort",
        effort,
    ]
    client = [
        str(sim_python),
        "-u",
        "-m",
        "roboicl.robodojo",
        "--task",
        task,
        "--seed",
        str(args.seed),
        "--layout",
        str(args.layout),
        "--data-root",
        str(data_root),
        "--results-root",
        str(results_root),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--enable_cameras",
        "--headless",
        "--kit_args",
        "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
    ]
    if args.capture_only:
        client.append("--capture-only")
        server = []
    record = {
        "arguments": vars(args),
        "robodojo_task": task,
        "layout_file": str(layout_path),
        "layout_sha256": sha(layout_path),
        "harness": config,
        "demo_manifest": env.get("ASTRA_DEMO_MANIFEST"),
        "server_command": server,
        "client_command": client,
        "output": str(out),
        "data_root": str(data_root),
        "results_root": str(results_root),
        "upstream_lock_sha256": sha(CODE / "configs/upstream.lock.json") if (CODE / "configs/upstream.lock.json").is_file() else None,
        "sim_python": str(sim_python),
        "policy_python": str(policy_python),
        "conda_exe": str(conda_exe) if conda_exe else None,
        "harness_profile": profile,
        "harness_profile_file": str(profile_path),
        "harness_profile_sha256": sha(profile_path),
        "provider_api_mode": api_mode,
        "provider_endpoint": env.get("ASTRA_ENDPOINT"),
        "provider_stream": env["ASTRA_STREAM"] == "1",
        "provider_http_backend": "python-urllib",
        "source_manifest_sha256": sha(CODE / "RELEASE_MANIFEST.json") if (CODE / "RELEASE_MANIFEST.json").is_file() else None,
        "reference_preflight": preflight_report,
        "asset_preflight": asset_report,
    }
    if args.dry_run:
        record["upstream"] = verify_submodules()
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    record["upstream"] = verify_submodules()

    if args.capture_only:
        env.update(
            ASTRA_CAMERA_PREFLIGHT=str(out / "capture"),
        )

    if not args.capture_only:
        for k in ["ASTRA_API_KEY", "ASTRA_ENDPOINT"]:
            if not env.get(k):
                raise ValueError(f"Supply {k} externally")

    results_root.mkdir(parents=True, exist_ok=True)
    gpu_slot, lock = acquire_gpu_slot(args.gpu, args.gpu_slot, results_root=results_root)
    record["gpu_slot"] = gpu_slot

    out.mkdir()
    (out / "harness.json").write_text(json.dumps(config, indent=2))
    record["code_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (out / "code_diff.patch").write_text(
        subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=CODE,
            capture_output=True,
            text=True,
        ).stdout
    )
    if args.shots:
        record["demo_sha256"] = sha(ref / "train_reference_bundle.json")
    (out / "run.json").write_text(json.dumps(record, indent=2))

    def wrapped(interpreter, cmd):
        if conda_exe is None:
            return cmd
        return [str(conda_exe), "run", "--no-capture-output", "--prefix",
                str(interpreter.parent.parent), *cmd]

    childenv = env.copy()
    childenv["PYTHONPATH"] = os.pathsep.join(python_paths("sim"))
    env["PATH"] = str(policy_python.parent) + os.pathsep + env.get("PATH", "")
    childenv["PATH"] = str(sim_python.parent) + os.pathsep + childenv.get("PATH", "")
    # Some Isaac Sim installations need an explicit libstdc++ preload; leave
    # the choice to the installation instead of assuming a Conda layout.
    if os.environ.get("ROBOICL_SIM_LD_PRELOAD"):
        childenv["LD_PRELOAD"] = os.environ["ROBOICL_SIM_LD_PRELOAD"]
    icd = "/etc/vulkan/icd.d/nvidia_icd.json"
    if Path(icd).exists():
        childenv.update(VK_DRIVER_FILES=icd, VK_ICD_FILENAMES=icd)

    p = q = None
    status = "failed"
    rc = 1
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    print(f"Run: {out}", flush=True)

    try:
        if not args.capture_only:
            with (out / "server.log").open("w") as log:
                p = subprocess.Popen(
                    wrapped(policy_python, server),
                    cwd=CODE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            deadline = time.monotonic() + 300
            while True:
                if p.poll() is not None:
                    raise RuntimeError(
                        f"Policy exited {p.returncode}; "
                        f"see {out}/server.log"
                    )
                try:
                    policy_socket_ready(port)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy startup exceeded 300 seconds")
                    time.sleep(1)

        with (out / "client.log").open("w") as log:
            q = subprocess.Popen(
                wrapped(sim_python, client),
                cwd=CODE,
                env=childenv,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        while q.poll() is None:
            if p is not None and p.poll() is not None:
                # Let the evaluator receive the RPC disconnect and score eligible state.
                try:
                    q.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        "Policy server terminated; evaluator did not finalize within "
                        "60 seconds"
                    )
            time.sleep(1)

        rc = q.returncode
        record["client_returncode"] = rc
        status = "client_exited" if rc == 0 else "failed"
        if args.capture_only:
            probe = out / "capture/capture_probe.json"
            if probe.exists():
                rows = json.loads(probe.read_text())
                observed = [r for r in rows if r["route"] == "observation"]
                if len(observed) != 30 or any(
                    r["max"] == 0 or r["shape"] != [480, 640, 3]
                    for r in observed
                ):
                    raise RuntimeError("Camera probe invalid")
                rc = 0
                status = "capture_complete_no_api"
            else:
                raise RuntimeError("Capture-only run did not produce camera probe")
        elif rc == 0:
            result_path = (
                results_root
                / "eval_result/RoboDojo"
                / task
                / "RoboICL/arx_x5"
                / f"{args.seed}_action_type=ee"
                / runid
                / "_result.json"
            )
            if not result_path.exists():
                raise RuntimeError(
                    "Client exited without scored result; see client.log"
                )
            result = json.loads(result_path.read_text())
            details = list(result.get("details", {}).values())
            if len(details) != 1 or details[0].get("layout_id") != args.layout:
                raise RuntimeError(
                    "Scored result does not match the selected single layout"
                )
            (out / "result.json").write_text(json.dumps(result, indent=2))
            status = "scored_with_error" if result.get("policy_error") else "scored"
    except KeyboardInterrupt:
        status = "interrupted"
        rc = 130
    except Exception:
        status = "failed"
        rc = 1
        raise
    finally:
        stop(q)
        stop(p)
        (out / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "returncode": rc,
                    "note": "Process exit is not task success; read evaluator scores.",
                },
                indent=2,
            )
        )

    print(f"{status}: {out}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
