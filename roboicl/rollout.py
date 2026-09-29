"""RoboICL's single-environment execution loop for an upstream RoboDojo env."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from roboicl.policy.action_feedback import TrackingGuard, tracking_receipt
from roboicl.policy.bounded_policy import MAX_RETRY_AFTER_S
from roboicl.policy.camera_readiness import blank_cameras
from roboicl.policy.dialogue_policy import DialogueConfig, load_config
from roboicl.policy.eval_failure import PolicyServiceError, policy_call
from roboicl.policy.head_calibration_capture import capture_head_calibration


def _rpc_timeout_s(config: DialogueConfig) -> float:
    retry_delay = sum(
        max(min(2 ** (attempt - 1), 8), MAX_RETRY_AFTER_S)
        for attempt in range(1, config.max_transport_attempts)
    )
    return (config.max_action_corrections + 1) * (
        config.max_transport_attempts * (config.request_timeout_s + 2) + retry_delay
    ) + 60


def _plain(action: dict) -> dict:
    return {key: np.asarray(value).tolist() for key, value in action.items()}


def _ready_observation(env):
    attempts = []
    for attempt in range(31):
        observation = env.get_obs()
        bad = blank_cameras({0: observation})
        if not bad:
            return observation
        attempts.append({"attempt": attempt, "blank_cameras": bad})
    env._policy_failure = {
        "method": "camera_capture",
        "classification": "camera_capture_failure",
        "official_score": None,
        "attempts": attempts,
    }
    raise PolicyServiceError("Camera capture remained blank after 30 render-only retries")


def eval_one_episode(env, model_client) -> None:
    """Execute only policy-selected actions and return observations after each step."""
    config = load_config(os.environ.get("ASTRA_HARNESS_CONFIG"))
    # The EE policy produces poses, but the simulator dispatches the IK result.
    # Install the execution-side guard in the same process that solves IK so a
    # discontinuous target is rejected before interpolation or queue mutation.
    env.ik_continuity_guard = None
    if config.ik_continuity_guard:
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("IK continuity guard requires one environment per Astra session")
        from src.eval_client.ik_continuity import IKContinuityError, IKContinuityGuard

        env.ik_continuity_guard = IKContinuityGuard(
            config.max_joint_increment_rad,
            log_path=Path(env.save_dir) / "_ik_continuity.jsonl",
        )
    if hasattr(model_client, "_client"):
        model_client._client.config.request_timeout_s = _rpc_timeout_s(config)
    policy_call(env, model_client, "reset")
    step = 0
    turns = 0
    observation = _ready_observation(env)
    observation["astra_meta"] = {
        "step": step,
        "max_episode_steps": int(env.step_lim),
    }
    if config.include_head_calibration:
        observation["astra_meta"]["head_calibration_context"] = capture_head_calibration(
            env, observation
        )
    policy_call(env, model_client, "update_obs", obs=observation)

    while not env.is_episode_end():
        if config.max_turns is not None and turns >= config.max_turns:
            raise PolicyServiceError("Harness action dialogue reached max_turns")
        actions = policy_call(env, model_client, "get_action")
        turns += 1
        tracking = TrackingGuard() if os.environ.get("ASTRA_TRACKING_GUARD", "1") == "1" else None
        for action_index, action in enumerate(actions):
            if env.is_episode_end():
                break
            if env.ik_continuity_guard is None:
                env.take_action(action)
            else:
                try:
                    env.take_action(action)
                except IKContinuityError as exc:
                    # The rejected proposal consumed no simulator step. Tell
                    # the policy about the exact rejected index so it can
                    # commit a zero-step receipt and replan from the same
                    # observation.
                    event = dict(
                        exc.event,
                        action_index=action_index,
                        discarded_commands=len(actions) - action_index,
                    )
                    policy_call(
                        env,
                        model_client,
                        "notify_action_rejection",
                        obs={"step": step, "event": event},
                    )
                    break
            step += 1
            observation = _ready_observation(env)
            observation["astra_meta"] = {
                "step": step,
                "max_episode_steps": int(env.step_lim),
                "executed_action": _plain(action),
            }
            interrupted = tracking is not None and tracking.observe(
                tracking_receipt(action, observation["state"])
            )
            if interrupted and action_index + 1 < len(actions):
                observation["astra_meta"]["controller_event"] = {
                    "type": "chunk_interrupted_tracking_error",
                    "discarded_commands": len(actions) - action_index - 1,
                    "reason": "Two consecutive large arm target residuals; reobserve and replan.",
                }
            policy_call(env, model_client, "update_obs", obs=observation)
            if interrupted:
                break
    policy_call(env, model_client, "finalize")
