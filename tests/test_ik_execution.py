"""Offline regression tests for the current RoboICL IK rejection path.

These tests intentionally load the checked-out RoboDojo dispatch method and
the current ``roboicl.rollout`` module.  They do not import Isaac Sim, start a
policy server, call an API, or require benchmark assets.
"""

from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBODOJO = ROOT / "third_party" / "RoboDojo"
sys.path.insert(0, str(ROBODOJO))

from src.eval_client.ik_continuity import (  # noqa: E402
    IKContinuityError,
    IKContinuityGuard,
)


def _dispatch_method():
    """Compile the current nested ``EvalEnv.take_action_batch`` method."""
    source = ROBODOJO / "src" / "eval_client" / "eval_env.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "take_action_batch"
    )
    namespace = {
        "deepcopy": deepcopy,
        "np": np,
        "IKContinuityError": IKContinuityError,
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["take_action_batch"]


def _ee_action(left_gripper=0.4, right_gripper=0.5):
    return {
        "left_ee_pose": [0.001, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "left_ee_joint_state": [left_gripper],
        "right_ee_pose": [0.001, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "right_ee_joint_state": [right_gripper],
    }


def _joint_action(value=0.02, left_gripper=0.4, right_gripper=0.5):
    joints = [float(value)] * 6
    return {
        "left_arm_joint_state": joints[:],
        "left_ee_joint_state": [left_gripper],
        "right_arm_joint_state": joints[:],
        "right_ee_joint_state": [right_gripper],
    }


def _dispatch_env(targets, current=None, guard=True):
    current = dict(current or {"left": [0.0] * 6, "right": [0.0] * 6})
    targets = dict(targets)
    robots = [
        SimpleNamespace(
            type="target",
            robot_name="x5",
            arm_name=f"{side}_arm",
            gripper_name=f"{side}_ee",
            ee_type="gripper",
            arm_joint_indices=list(range(6)),
            gripper_move={"sign": 1, "mimic": [0, 1, 0]},
            gripper_scale=[0, 1],
        )
        for side in ("left", "right")
    ]

    def get_joint(robot, **_kwargs):
        return {0: np.asarray(current[robot.arm_name.split("_")[0]], dtype=float).copy()}

    def solve_ik(*, robot, **_kwargs):
        side = robot.arm_name.split("_")[0]
        return {"status": "Success", "joint_value": np.asarray(targets[side], dtype=float).copy()}

    manager = SimpleNamespace(
        robot_list=robots,
        process_name=lambda name: f"{name}_joint_state",
        get_joint=get_joint,
        get_joint_limits=lambda robot, **_kwargs: {0: np.asarray([[-10.0, 10.0]] * 6)},
        solve_ik=Mock(side_effect=solve_ik),
        control_manager=SimpleNamespace(push=Mock()),
    )
    return SimpleNamespace(
        physx_monitor_enabled=False,
        num_envs=1,
        robot_manager=manager,
        take_action_cnt=[0],
        step_lim=20,
        end_flag=[False],
        ik_continuity_guard=IKContinuityGuard(0.15) if guard else None,
        validate_action_dict=Mock(),
        get_action_type=lambda action: (
            "ee" if "left_ee_pose" in action else "joint"
        ),
        process_control_info=Mock(side_effect=lambda info, _env_idx: info),
        have_empty=lambda _env_ids: True,
        step=Mock(),
        reward_manager=SimpleNamespace(step=Mock()),
        is_episode_end=Mock(return_value=False),
    )


class CurrentDispatchTests(unittest.TestCase):
    def test_sample_steps_excludes_endpoints_when_requested(self):
        from roboicl.policy.dialogue_policy import sample_steps

        # Interior-only sampling must never return the first or last frame.
        self.assertEqual(sample_steps([0, 1, 2, 3, 4], 3, include_endpoints=False), [1, 2, 3])
        self.assertEqual(sample_steps([0, 1, 2, 3, 4, 5], 4, include_endpoints=False), [1, 2, 3, 4])
        # Endpoint inclusion is unchanged.
        self.assertEqual(sample_steps([0, 1, 2, 3, 4], 3, include_endpoints=True), [0, 2, 4])
        # With only two observed frames there is no interior to sample.
        self.assertEqual(sample_steps([0, 1], 2, include_endpoints=False), [])
        # More interior than needed keeps the full interior, never the endpoints.
        self.assertEqual(sample_steps([0, 1, 2], 3, include_endpoints=False), [1])
        self.assertEqual(sample_steps([0, 1, 2, 3], 4, include_endpoints=False), [1, 2])

    @classmethod
    def setUpClass(cls):
        cls.dispatch = _dispatch_method()

    def test_any_arm_jump_rejects_atomically_before_queue_or_step(self):
        env = _dispatch_env({"left": [0.05] * 6, "right": [0.30] * 6})
        action = _ee_action()
        original = deepcopy(action)

        with self.assertRaises(IKContinuityError) as caught:
            type(self).dispatch(env, [action], env_idx_list=[0])

        self.assertEqual(caught.exception.event["type"], "action_rejected_ik_continuity")
        self.assertFalse(caught.exception.event["rejected_action_executed"])
        self.assertEqual(env.take_action_cnt, [0])
        env.robot_manager.control_manager.push.assert_not_called()
        env.process_control_info.assert_not_called()
        env.step.assert_not_called()
        env.reward_manager.step.assert_not_called()
        self.assertEqual(env.ik_continuity_guard.previous_targets, {})
        self.assertEqual(action, original)

    def test_accepted_target_is_normalized_before_dispatch(self):
        env = _dispatch_env({
            "left": [0.05, 0.05, 2 * np.pi + 0.10, 0.05, 0.05, 0.05],
            "right": [0.10] * 6,
        })

        type(self).dispatch(env, [_ee_action()], env_idx_list=[0])

        self.assertEqual(env.take_action_cnt, [1])
        env.robot_manager.control_manager.push.assert_called_once()
        info = env.process_control_info.call_args.args[0]
        self.assertAlmostEqual(info["left_arm_joint_state"]["position"][2], 0.10)
        self.assertAlmostEqual(
            env.ik_continuity_guard.previous_targets["left_arm_joint_state"][2], 0.10
        )

    def test_previous_accepted_target_jump_is_also_atomic(self):
        env = _dispatch_env({"left": [0.10] * 6, "right": [0.10] * 6})
        type(self).dispatch(env, [_ee_action()], env_idx_list=[0])
        env.take_action_cnt[:] = [1]
        env.robot_manager.control_manager.push.reset_mock()
        env.process_control_info.reset_mock()
        env.robot_manager.solve_ik.side_effect = lambda *, robot, **_kwargs: {
            "status": "Success",
            "joint_value": np.asarray([0.30] * 6),
        }

        with self.assertRaises(IKContinuityError):
            type(self).dispatch(env, [_ee_action()], env_idx_list=[0])

        self.assertEqual(env.take_action_cnt, [1])
        env.robot_manager.control_manager.push.assert_not_called()
        env.process_control_info.assert_not_called()
        np.testing.assert_allclose(
            env.ik_continuity_guard.previous_targets["left_arm_joint_state"], [0.10] * 6
        )

    def test_malformed_ik_candidate_fails_closed_as_continuity_rejection(self):
        guard = IKContinuityGuard(0.15)
        with self.assertRaises(IKContinuityError) as caught:
            guard.check({"left_arm_joint_state": None})
        self.assertEqual(caught.exception.checks[0]["error"], "invalid_candidate")
        self.assertFalse(caught.exception.event["rejected_action_executed"])
        self.assertEqual(guard.previous_targets, {})

    def test_dispatch_build_failure_does_not_commit_unexecuted_target(self):
        env = _dispatch_env({"left": [0.05] * 6, "right": [0.05] * 6})
        env.process_control_info.side_effect = RuntimeError("offline interpolation failure")
        with self.assertRaisesRegex(RuntimeError, "interpolation failure"):
            type(self).dispatch(env, [_ee_action()], env_idx_list=[0])
        self.assertEqual(env.take_action_cnt, [0])
        self.assertEqual(env.ik_continuity_guard.previous_targets, {})
        env.robot_manager.control_manager.push.assert_not_called()

    def test_joint_hold_refreshes_baseline_for_next_ee_action(self):
        env = _dispatch_env({"left": [0.10] * 6, "right": [0.10] * 6})
        type(self).dispatch(env, [_ee_action()], env_idx_list=[0])

        env.robot_manager.solve_ik.side_effect = lambda *, robot, **_kwargs: {
            "status": "Success",
            "joint_value": np.asarray([0.12] * 6),
        }
        type(self).dispatch(env, [_joint_action(0.02)], env_idx_list=[0])
        type(self).dispatch(env, [_ee_action()], env_idx_list=[0])

        self.assertEqual(env.take_action_cnt, [3])
        np.testing.assert_allclose(
            env.ik_continuity_guard.previous_targets["left_arm_joint_state"], [0.12] * 6
        )
        env.robot_manager.control_manager.push.assert_called()


class CurrentRolloutTests(unittest.TestCase):
    def test_rejection_drops_suffix_and_notifies_current_rollout(self):
        scipy = types.ModuleType("scipy")
        scipy_spatial = types.ModuleType("scipy.spatial")
        scipy_transform = types.ModuleType("scipy.spatial.transform")
        scipy_transform.Rotation = SimpleNamespace()
        scipy.spatial = scipy_spatial
        scipy_spatial.transform = scipy_transform
        with patch.dict(sys.modules, {
            "scipy": scipy,
            "scipy.spatial": scipy_spatial,
            "scipy.spatial.transform": scipy_transform,
        }):
            from roboicl import rollout

        config = SimpleNamespace(
            ik_continuity_guard=True,
            max_joint_increment_rad=0.15,
            max_transport_attempts=1,
            request_timeout_s=1,
            max_action_corrections=0,
            max_turns=None,
            include_head_calibration=False,
        )
        actions = [_ee_action(), _ee_action(), _ee_action()]
        events = []
        save_dir = tempfile.TemporaryDirectory(prefix="roboicl-ik-rollout-")
        self.addCleanup(save_dir.cleanup)

        class Env:
            num_envs = 1
            step_lim = 10

            def __init__(self):
                self.save_dir = save_dir.name
                self.done = False
                self.attempted = []
                self.executed = []

            def is_episode_end(self):
                return self.done

            def take_action(self, action):
                self.attempted.append(action)
                if len(self.attempted) == 2:
                    guard = IKContinuityGuard(0.15)
                    try:
                        guard.check({
                            "right_arm_joint_state": {
                                "status": "Success",
                                "current": [0.0] * 6,
                                "target": [0.3] * 6,
                                "joint_limits": [[-10.0, 10.0]] * 6,
                                "periodic": False,
                            }
                        })
                    except IKContinuityError as exc:
                        raise exc
                self.executed.append(action)

        env = Env()

        def fake_policy_call(_env, _client, func_name, **kwargs):
            events.append((func_name, kwargs.get("obs")))
            if func_name == "get_action":
                return actions
            if func_name == "notify_action_rejection":
                env.done = True
            return None

        observation = {
            "state": {},
            "vision": {},
            "instruction": "offline",
        }
        with patch.dict(os.environ, {"ASTRA_TRACKING_GUARD": "0"}, clear=False), \
             patch.object(rollout, "load_config", return_value=config), \
             patch.object(rollout, "policy_call", side_effect=fake_policy_call), \
             patch.object(rollout, "_ready_observation", return_value=deepcopy(observation)):
            rollout.eval_one_episode(env, object())

        self.assertEqual(len(env.executed), 1)
        self.assertEqual(len(env.attempted), 2)
        notifications = [item for item in events if item[0] == "notify_action_rejection"]
        self.assertEqual(len(notifications), 1)
        event = notifications[0][1]["event"]
        self.assertEqual(event["action_index"], 1)
        self.assertEqual(event["discarded_commands"], 2)
        self.assertEqual(notifications[0][1]["step"], 1)
        self.assertEqual(events[-1][0], "finalize")

    def test_zero_step_rejection_receipt_uses_current_model(self):
        from roboicl.policy.dialogue_policy import Model

        row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        response = {
            "output": [{
                "type": "function_call",
                "name": "act",
                "arguments": json.dumps({
                    "actions": [row, row, row],
                    "execution_note": "offline rejection fixture",
                }),
                "call_id": "blocked",
            }]
        }
        with tempfile.TemporaryDirectory(prefix="roboicl-ik-policy-") as directory:
            config_path = Path(directory) / "harness.json"
            config_path.write_text(json.dumps({
                "predict_horizon": 3,
                "execute_horizon": 3,
                "feedback_frames": 1,
                "ik_continuity_guard": True,
                "max_action_corrections": 2,
            }), encoding="utf-8")
            env = {
                "ASTRA_API_KEY": "offline",
                "ASTRA_LOG_DIR": directory,
                "ASTRA_DEMO_MANIFEST": "",
                "ASTRA_HARNESS_CONFIG": str(config_path),
            }
            with patch.dict(os.environ, env, clear=False):
                model = Model({})
                obs = {
                    "instruction": "offline",
                    "astra_meta": {"step": 0, "max_episode_steps": 10},
                    "state": {
                        "left_ee_pose": [1, 2, 3, 1, 0, 0, 0],
                        "right_ee_pose": [0, 0, 1, 1, 0, 0, 0],
                        "left_arm_joint_state": [0] * 6,
                        "right_arm_joint_state": [0] * 6,
                        "left_ee_joint_state": [0.4],
                        "right_ee_joint_state": [0.5],
                    },
                    "vision": {
                        name: {"color": np.full((8, 8, 3), 1, dtype=np.uint8)}
                        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
                    },
                }
                model.update_obs(obs)
                with patch.object(model, "_request", return_value=response):
                    self.assertEqual(len(model.get_action()), 3)
                event = {
                    "type": "action_rejected_ik_continuity",
                    "rejected_action_executed": False,
                    "action_index": 0,
                    "discarded_commands": 3,
                    "violations": [{
                        "arm": "right_arm_joint_state",
                        "error": "joint_target_jump",
                        "max_current_delta_rad": 0.3,
                        "limit_rad": 0.15,
                    }],
                }
                model.notify_action_rejection({"step": 0, "event": event})
                payload = model.build_payload()

                output = next(
                    item for item in payload["input"]
                    if item.get("type") == "function_call_output"
                    and item.get("call_id") == "blocked"
                )
                receipt = json.loads(output["output"])
                self.assertEqual(receipt["executed_step_interval"], [0, 0])
                self.assertEqual(receipt["executed_steps"], 0)
                self.assertEqual(receipt["unexecuted_steps"], 3)
                self.assertEqual(receipt["executed_prediction_indices_inclusive"], [])
                self.assertEqual(receipt["controller_events"][0]["type"],
                                 "action_rejected_ik_continuity")
                self.assertEqual(model.ledger, [])
                self.assertIsNone(model.pending)

if __name__ == "__main__":
    unittest.main()
