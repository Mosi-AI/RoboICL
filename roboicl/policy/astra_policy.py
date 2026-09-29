"""Observation-only RoboDojo policy; derived from the imported Responses harness.

No checkpoint, expert trajectory, task implementation or simulator object lookup
is used. Each episode owns its visual archive, executed-action ledger and memory.
"""
from __future__ import annotations

import base64
import json
import os
import http.client
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from .action_feedback import tracking_receipt
from .history_context import recent_context, retrieve_records
from .provider_compat import from_wire_response, to_wire_payload, validate_api_mode, validate_endpoint

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = (
    "left_arm_joint_state", "left_ee_joint_state", "left_ee_pose",
    "right_arm_joint_state", "right_ee_joint_state", "right_ee_pose",
)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def image_url(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("Camera must supply an HWC RGB/RGBA array")
    buf = BytesIO()
    Image.fromarray(array[:, :, :3].astype(np.uint8)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def validate_actions(raw, max_chunk=32, state=None):
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_chunk:
        raise ValueError(f"actions must contain 1..{max_chunk} control steps")
    result = []
    ee_targets = {side: np.asarray((state or {}).get(f"{side}_ee_pose", []), dtype=np.float64).copy() for side in ("left", "right")}
    for item in raw:
        if item.get("mode") not in ("joint", "ee", "ee_delta"):
            raise ValueError("mode must be joint, ee or ee_delta")
        mode = item["mode"]
        size = 7 if mode == "ee" else 6
        packed = {}
        for side in ("left", "right"):
            values = np.asarray(item[side], dtype=np.float32)
            grip = item[f"{side}_gripper"]
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"{side} requires {size} finite numbers for {mode}")
            if isinstance(grip, bool) or not isinstance(grip, (int, float)) or not np.isfinite(grip) or not 0 <= grip <= 1:
                raise ValueError("gripper commands must be finite values in [0,1]")
            if mode == "ee_delta":
                previous = ee_targets[side]
                if previous.shape != (7,) or not np.isfinite(previous).all():
                    raise ValueError("ee_delta requires an observed EE pose; do not mix joint then delta in one chunk")
                angle = float(np.linalg.norm(values[3:]))
                dq = np.r_[np.cos(angle/2), values[3:] * (np.sin(angle/2)/angle if angle > 1e-10 else 0.5)]
                # World-frame incremental rotation multiplies on the left.
                values = np.r_[previous[:3] + values[:3], _quat_mul(dq, previous[3:])].astype(np.float32)
            if mode in ("ee", "ee_delta"):
                norm = float(np.linalg.norm(values[3:]))
                if norm < 1e-6 or abs(norm - 1) > 0.05:
                    raise ValueError("EE quaternion must be unit length, ordered w,x,y,z")
                values[3:] /= norm
                ee_targets[side] = values.copy()
            else:
                ee_targets[side] = np.array([])
            key = f"{side}_arm_joint_state" if mode == "joint" else f"{side}_ee_pose"
            packed[key] = values
            packed[f"{side}_ee_joint_state"] = np.asarray([grip], dtype=np.float32)
        result.append(packed)
    return result


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.api_mode = validate_api_mode(
            os.environ.get("ASTRA_API_MODE") or model_cfg.get("api_mode", "responses"))
        self.endpoint = validate_endpoint(
            self.api_mode,
            os.environ.get("ASTRA_ENDPOINT") or model_cfg.get("endpoint")
            or "https://api.openai.com/v1/responses",
        )
        self.model_name = model_cfg.get("model", "gpt-6-astra")
        self.api_key = os.environ.get("ASTRA_API_KEY")
        if not self.api_key:
            raise RuntimeError("ASTRA_API_KEY is required (use the private credential launcher)")
        self.effort = os.environ.get("ASTRA_REASONING_EFFORT", model_cfg.get("reasoning_effort", "xhigh"))
        self.max_chunk = int(model_cfg.get("max_chunk", 32))
        configured_output = model_cfg.get("max_output_tokens")
        self.max_output = int(configured_output) if configured_output is not None else None
        self.max_rounds = int(model_cfg.get("max_tool_rounds", 16))
        self.retrievable_history = os.environ.get("ASTRA_RETRIEVABLE_HISTORY", "0") == "1"
        self.log_root = Path(os.environ.get("ASTRA_LOG_DIR", "eval_result/astra_traces"))
        self.reset()

    def reset(self):
        self.archive = {}
        self.head_calibration_context = None
        self.train_reference_checks = None
        self.ledger = []
        self.decisions = []
        self.memory = ""
        self.step = None
        self.episode_id = uuid4().hex
        self.episode_dir = self.log_root / self.episode_id

    def _log(self, name, value):
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        with (self.episode_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + "\n")

    def update_obs(self, obs):
        meta = obs.get("astra_meta", {})
        step = int(meta.get("step", len(self.archive)))
        if step in self.archive:
            self.step = step
            return
        if meta.get('head_calibration_context') is not None:
            if self.head_calibration_context is not None or self.archive:
                raise ValueError('Head calibration can only initialize an empty episode')
            self.head_calibration_context = plain(meta['head_calibration_context'])
            self._log('head_calibration_context.jsonl', self.head_calibration_context)
            # DialogueModel may have loaded a persisted verification sidecar
            # for a portable bundle whose original HDF5 is not present.  Do
            # not discard that checked report and attempt an unavailable
            # source re-read on the first live observation.
            if getattr(self, 'demo', None) and self.train_reference_checks is None:
                from .train_reference_checks import verify_training_manifest
                self.train_reference_checks = verify_training_manifest(self.demo)
                self._log('train_reference_checks.jsonl', self.train_reference_checks)
        # Whitelist proprioception and RGB only: no object truth or rewards.
        entry = {
            "step": step,
            "state": {k: plain(obs.get("state", {})[k]) for k in STATE_KEYS if k in obs.get("state", {})},
            "instruction": str(obs.get("instruction", "")),
            "images": {name: image_url(obs["vision"][name]["color"]) for name in CAMERAS},
        }
        if meta.get('max_episode_steps') is not None:
            entry['max_episode_steps'] = int(meta['max_episode_steps'])
        self.archive[step] = entry
        self.step = step
        if meta.get("executed_action") is not None:
            submitted = plain(meta["executed_action"])
            self.ledger.append({"resulting_step": step, "executed_action": submitted,
                                "state_after": entry["state"],
                                "tracking": tracking_receipt(submitted, entry["state"])})
            if meta.get("controller_event"):
                self.ledger[-1]["controller_event"] = plain(meta["controller_event"])
        self._log("observations.jsonl", entry)

    def update_obs_batch(self, obs):
        if len(obs) != 1:
            raise ValueError("Use one Astra policy session per environment")
        self.update_obs(obs[0])

    def get_action_batch(self, env_idx_list=None):
        return [self.get_action()]

    def _visual_content(self, steps):
        content = []
        for step in steps:
            if step not in self.archive:
                raise ValueError(f"Observation step {step} not available")
            entry = self.archive[step]
            content.append({"type": "input_text", "text": json.dumps({"observation_step": step, "state": entry["state"]})})
            for name, url in entry["images"].items():
                content.extend([
                    {"type": "input_text", "text": f"Step {step}, camera {name}"},
                    {"type": "input_image", "image_url": url, "detail": "high"},
                ])
        return content

    def _tools(self):
        tools = [
            {"type": "function", "name": "inspect_history", "description": "Retrieve any observed steps from this episode, including all three images and robot state. Use to recover earlier cues or compare action effects.", "parameters": {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 8}}, "required": ["steps"], "additionalProperties": False}},
            {"type": "function", "name": "act", "description": f"Execute 1..{self.max_chunk} target commands then reobserve/replan. Save your complete ongoing memory for the next decision.", "parameters": {"type": "object", "properties": {
                "memory": {"type": "string", "description": "Persistent facts, vanished cues, progress, uncertainties, plan and useful historical step IDs."},
                "actions": {"type": "array", "minItems": 1, "maxItems": self.max_chunk, "items": {"type": "object", "properties": {
                    "mode": {"type": "string", "enum": ["ee_delta", "ee", "joint"]},
                    "left": {"type": "array", "items": {"type": "number"}},
                    "right": {"type": "array", "items": {"type": "number"}},
                    "left_gripper": {"type": "number", "minimum": 0, "maximum": 1},
                    "right_gripper": {"type": "number", "minimum": 0, "maximum": 1},
                }, "required": ["mode", "left", "right", "left_gripper", "right_gripper"], "additionalProperties": False}},
            }, "required": ["memory", "actions"], "additionalProperties": False}},
        ]

        if self.retrievable_history:
            tools.append({"type": "function", "name": "inspect_records",
                          "description": "Retrieve exact earlier executed actions, resulting robot states and model decision notes. Inclusive range, at most32 steps. No images or success truth; use inspect_history for images. result step s follows execution of its recorded action.",
                          "parameters": {"type": "object", "properties": {
                              "start_step": {"type": "integer", "minimum": 0},
                              "end_step": {"type": "integer", "minimum": 0}},
                              "required": ["start_step", "end_step"], "additionalProperties": False}})
        return tools

    def _request(self, payload):
        wire_payload = to_wire_payload(payload, self.api_mode)
        authorization = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(wire_payload).encode("utf-8"),
            headers={"Authorization": authorization,
                     "Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    result = from_wire_response(json.load(response), self.api_mode)
                break
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                retryable = code in (408, 429, 500, 502, 503, 504)
                failure = f"HTTP {code}"
            except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError, ConnectionError):
                retryable = True
                failure = "transport_error"
            # No raw exception/response body: either may echo credentials.
            self._log("transport.jsonl", {"step": self.step, "attempt": attempt + 1,
                      "failure": failure, "retrying": retryable and attempt < 2,
                      "robot_actions_executed_by_request": 0,
                      "provider_may_have_processed_request": True})
            if not retryable or attempt == 2:
                raise RuntimeError(f"Astra API {failure}; no robot action dispatched") from None
            # Retrying inference may incur duplicate provider charges, but tools
            # are local and no command is dispatched until a response is accepted.
            time.sleep(2 ** attempt)
        if result.get("error"):
            raise RuntimeError("Astra API returned an error response")
        return result

    def get_action(self):
        raise RuntimeError(
            "The legacy Astra policy entrypoint was removed; use "
            "roboicl.policy.dialogue_policy.Model"
        )
