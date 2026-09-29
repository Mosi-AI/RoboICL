"""Fail-closed policy RPC handling and preservation, without model API access."""
import json
import os
from pathlib import Path
import shutil


class PolicyServiceError(RuntimeError):
    pass


def policy_call(env, client, func_name, **kwargs):
    try:
        return client.call(func_name=func_name, **kwargs)
    except Exception as exc:
        # Do not persist raw exception bodies: providers can echo sensitive input.
        env._policy_failure = {"method": func_name, "exception_type": type(exc).__name__,
                               "classification": "policy_service_failure",
                               "official_score": None, "error_code": safe_error_code(exc)}
        raise PolicyServiceError(f"Policy RPC {func_name} failed; evaluation must stop") from None


def preserve_failure(env):
    root = Path(env.save_dir)
    destination = root / "interrupted_videos"
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(env._policy_failure)
    report["videos"] = []
    report["video_errors"] = []
    # Remove writers from the normal abort/delete path even if finalization fails.
    for env_id in list(env.video_writers):
        writers = env.video_writers.pop(env_id)
        for camera, writer in writers.items():
            source = Path(writer.out_path)
            target = destination / f"env{env_id}_{camera}_interrupted.mp4"
            try:
                writer.close(announce=False)
                if target.exists():
                    raise FileExistsError("Preservation must not overwrite prior evidence")
                shutil.move(str(source), str(target))
                report["videos"].append(str(target))
            except Exception as exc:
                report["video_errors"].append({"camera": camera, "type": type(exc).__name__})
    path = root / "_infrastructure_failure.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def safe_error_code(exc):
    """Retain useful classification, never provider bodies or credentials."""
    explicit = getattr(exc, "error_code", None)
    if isinstance(explicit, str) and explicit:
        normalized = explicit.strip().lower()
        aliases = {
            "response.failed": "provider_failed",
            "response_failed": "provider_failed",
            "response.incomplete": "provider_incomplete",
            "response_incomplete": "provider_incomplete",
            "server_error": "provider_server_error",
            "service_unavailable": "provider_server_error",
            "overloaded": "provider_server_error",
            "rate_limit_exceeded": "rate_limit",
            "too_many_requests": "rate_limit",
            "timed_out": "timeout",
        }
        allowed = {
            "content_policy_violation", "invalid_encrypted_content",
            "invalid_comparison_response_id", "insufficient_quota",
            "rate_limit", "provider_server_error", "timeout",
            "policy_server_restarted", "action_rejected",
            "provider_incomplete", "provider_failed",
            "provider_stream_truncated", "connection_error",
            "policy_rpc_error",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in allowed:
            return normalized
    value = str(exc).lower()
    for code, markers in (
        ("content_policy_violation", ("content_policy_violation", "image violation")),
        ("invalid_encrypted_content", ("invalid_encrypted_content",)),
        ("invalid_comparison_response_id", ("comparison_response_id", "string_above_max_length")),
        ("insufficient_quota", ("insufficient_quota", "billing_hard_limit")),
        ("rate_limit", ("rate_limit", "429", "too many requests")),
        ("provider_server_error", ("provider_server_error", "server_error",
                                   "service_unavailable", "temporarily_unavailable",
                                   "provider_capacity", "overloaded")),
        ("timeout", ("timeout", "timed out")),
        ("policy_server_restarted", ("policy server restarted",)),
        ("action_rejected", ("action correction limit", "translation l2=", "rotation l2=",
                             "ik continuity")),
        ("provider_incomplete", ("provider_incomplete", "response.incomplete",
                                 "response_incomplete")),
        ("provider_failed", ("provider_failed", "response.failed", "response_failed")),
        ("provider_stream_truncated", ("provider_stream_truncated",
                                       "sse ended without response.completed",
                                       "sse done without response.completed")),
        ("connection_error", ("connection", "disconnect", "broken pipe")),
    ):
        if any(marker in value for marker in markers):
            return code
    return "policy_rpc_error"


def finalize_policy_failure(env):
    """Score an eligible healthy current state; never dispatch another action."""
    failure = getattr(env, "_policy_failure", None)
    minimum = int(os.environ.get("ROBODOJO_SCORE_POLICY_ERROR_MIN_STEPS", "0"))
    if minimum <= 0 or not failure or failure.get("classification") != "policy_service_failure":
        return False
    if failure.get("method") not in ("get_action", "update_obs", "finalize"):
        return False
    if env.num_envs != 1 or getattr(env, "unstable_envs", None):
        return False
    steps = int(env.take_action_cnt[0])
    if steps < minimum or getattr(env, "_scored_policy_failure", None):
        return False
    # Inspect an existing monitor only; do not create one for non-PhysX tasks.
    import sys
    for name, module in list(sys.modules.items()):
        if name.endswith("physx_warning_monitor") and hasattr(module, "get_monitor"):
            monitor = module.get_monitor()
            if monitor.is_fatal() or monitor.get_broken_envs():
                return False
    try:
        rewards = env.reward_manager.get_reward(final_check=True)
        env.success[0] = bool(rewards[0] > 1 - 1e-3)
        env.end_flag[0] = True
        env.get_obs_batch(env_idx_list=[0], last_frame=True)
    except Exception as exc:
        failure["final_score_error_type"] = type(exc).__name__
        return False
    env._scored_policy_failure = dict(failure, executed_steps=steps,
                                     minimum_steps=minimum,
                                     termination_reason="policy_service_error")
    env._scored_policy_failure.pop("official_score", None)
    return True
