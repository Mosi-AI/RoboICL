"""Thin adapter that runs one exact layout against the clean RoboDojo submodule."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import traceback
import types

from roboicl.paths import ROBODOJO, configure_imports

configure_imports("sim")

from isaaclab.app import AppLauncher


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--task", required=True)
    value.add_argument("--seed", required=True, type=int)
    value.add_argument("--layout", required=True, type=int)
    value.add_argument("--data-root", required=True, type=Path)
    value.add_argument("--results-root", required=True, type=Path)
    value.add_argument("--port", required=True, type=int)
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--capture-only", action="store_true")
    AppLauncher.add_app_launcher_args(value)
    return value


class _OfflineClient:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, *, func_name, **kwargs):
        if func_name != "reset":
            raise RuntimeError(f"Capture-only client does not implement {func_name}")
        return None

    def close(self):
        return None


def main() -> int:
    args = parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    assets = data_root / "Assets"
    layout = assets / "Eval_Layout/RoboDojo/arx_x5" / str(args.seed) / f"{args.task}_{args.layout}.json"
    if not layout.is_file():
        raise FileNotFoundError(layout)
    results_root.mkdir(parents=True, exist_ok=True)

    # ``env.global_configs`` snapshots these roots at import time.  The
    # launcher normally exports them, but this direct adapter is also a
    # supported entry point (for capture-only and one-layout debugging), so
    # set them before importing any RoboDojo runtime module.  Changing cwd is
    # not sufficient because EVAL_RESULT_PATH is absolute in global_configs.
    os.environ["ROBOICL_DATA_ROOT"] = str(data_root)
    os.environ["ROBOICL_RESULTS_ROOT"] = str(results_root)
    os.chdir(results_root)

    # Configure the clean submodule before any RoboDojo module copies these globals.
    global_configs = importlib.import_module("env.global_configs")
    global_configs.ASSETS_PATH = str(assets)
    global_configs.OBJECTS_PATH = str(assets / "Object/RoboDojo")
    global_configs.ROBOTS_PATH = str(assets / "Robots")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    stage = "import RoboDojo runtime"
    try:
        from omegaconf import OmegaConf
        eval_env_module = importlib.import_module("src.eval_client.eval_env")
        if args.capture_only:
            # EvalEnv constructs its transport in __init__, before the adapter
            # can replace env.model_client. Inject the no-op client at that one
            # owned boundary so capture-only never waits for a policy server.
            eval_env_module.WsModelClient = _OfflineClient
        create_eval_env = eval_env_module.create_eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        registry = importlib.import_module("task.RoboDojo.task_registry")
        stage = "assemble RoboDojo configuration"
        env_cfg_root = ROBODOJO / "env_cfg"
        base = load_yaml(env_cfg_root / "arx_x5.yml")
        eval_cfg = dict(base)
        eval_cfg.update(
            task_name=args.task,
            num_envs=1,
            device_id=0,
            eval_batch=False,
            policy_name="RoboICL",
            additional_info="action_type=ee",
            seed=args.seed,
            physx_monitor_enabled=False,
            eval_num=1,
        )
        deploy_cfg = {
            "policy_name": "RoboICL",
            "port": args.port,
            "host": args.host,
            "protocol": "ws",
            "policy_server_url": f"ws://{args.host}:{args.port}",
            "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
            "trial_id": f"{args.task}-{os.environ['ROBODOJO_RUN_ID']}",
            "action_case_id": f"{args.task}_case",
            "repeat_index": None,
        }
        task_config = registry.task_config_path(ROBODOJO / "task/RoboDojo/config", args.task)
        env_cfg = OmegaConf.create({
            "sim": load_yaml(env_cfg_root / "sim" / f"{eval_cfg['config']['sim']}.yml"),
            "scene": load_yaml(env_cfg_root / "scene" / f"{eval_cfg['config']['scene']}.yml"),
            "camera": load_yaml(env_cfg_root / "camera" / f"{eval_cfg['config']['camera']}.yml"),
            "robot": load_yaml(env_cfg_root / "robot" / f"{eval_cfg['config']['robot']}.yml"),
            "task_env": load_yaml(task_config),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        })
        env_cfg = process_randomization(env_cfg)
        env_cfg, _ = process_config(env_cfg, task_name=args.task)
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
        env_cfg.sim.seed = [0]
        OmegaConf.update(
            env_cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        stage = "create RoboDojo environment"
        env = create_eval_env(env_cfg, simulation_app)

        # Upstream currently enumerates matching files and may renumber gaps.
        # Replace that lookup with the exact public layout ID selected by the CLI.
        stage = "bind exact layout"
        env.seed_manager.seed_info = {args.layout: {"scene_layout": str(layout)}}
        env.seed_manager.seed_list = [args.layout]
        env.seed_manager.st_idx = 0
        env.seed_manager.ed_idx = 1
        env.seed_manager.idx = 0
        env.env_seeds = [args.layout]
        if args.capture_only:
            env.model_client = _OfflineClient()
        stage = "reset task scene"
        env.reset(seed=[args.layout])

        if args.capture_only:
            from roboicl.policy.camera_preflight import run
            stage = "capture camera probe"
            run(env)
            return 0

        from roboicl.rollout import eval_one_episode

        def owned_rollout(instance):
            return eval_one_episode(instance, instance.model_client)

        env.eval_one_episode = types.MethodType(owned_rollout, env)
        stage = "run evaluation"
        env.run_eval()
        return 0
    except BaseException:
        print(f"RoboICL adapter failed during: {stage}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            try:
                env.model_client.close()
            except Exception:
                pass
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
