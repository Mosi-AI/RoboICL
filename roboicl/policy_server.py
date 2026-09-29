"""Serve the owned RoboICL policy over XPolicyLab's pinned WebSocket transport."""

from __future__ import annotations

import argparse
import asyncio

from roboicl.paths import configure_imports

configure_imports()

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from roboicl.policy.dialogue_policy import Model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--reasoning-effort", default="xhigh")
    args = parser.parse_args()
    model = Model({
        "policy_name": "RoboICL",
        "model": args.model,
        "bench_name": "RoboDojo",
        "task_name": args.task,
        "seed": args.seed,
        "action_type": "ee",
        "max_chunk": 64,
        "max_tool_rounds": 16,
        "reasoning_effort": args.reasoning_effort,
    })
    server = PolicyServer(model, PolicyServerConfig(
        host=args.host,
        port=args.port,
        ws_ping_interval_s=30,
        ws_ping_timeout_s=600,
    ))
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
