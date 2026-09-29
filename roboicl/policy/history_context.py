"""Pure, opt-in context selection; does not change or execute a robot policy.

Callers retain complete episode-local archives. Record counts are bounded here,
not token counts: unusually large individual records still need size monitoring.
"""
from copy import deepcopy


def recent_context(*, task, current_step, available_steps, memory, ledger,
                   decisions, action_window=16, decision_window=4):
    if type(action_window) is not int or not 1 <= action_window <= 64:
        raise ValueError("action_window must be 1..64")
    if type(decision_window) is not int or not 1 <= decision_window <= 16:
        raise ValueError("decision_window must be 1..16")
    steps = sorted(available_steps)
    if not steps or current_step not in steps:
        raise ValueError("current_step must exist in the observation archive")
    actions = [r for r in ledger if r["resulting_step"] <= current_step]
    choices = [r for r in decisions if r["step"] <= current_step]
    events = [{"step": r["resulting_step"], "event": r["controller_event"]}
              for r in actions if "controller_event" in r]
    return deepcopy({
        "task": task, "current_step": current_step, "memory": memory,
        "available_history_steps": [steps[0], current_step],
        "decision_history": choices[-decision_window:],
        "executed_actions": actions[-action_window:],
        "latest_tracking_receipt": actions[-1].get("tracking", {}) if actions else {},
        "recent_controller_events": events[-8:],
        "history_scope": {
            "full_archive_retained": True,
            "omitted_actions": max(0, len(actions) - action_window),
            "omitted_decisions": max(0, len(choices) - decision_window),
            "omitted_controller_events": max(0, len(events) - 8),
            "retrieval": "Request earlier records by inclusive observation step range.",
            "memory_is_model_report_not_verified_truth": True,
        },
    })


def retrieve_records(*, start_step, end_step, current_step, ledger, decisions):
    """Return exact recorded actions/states/decisions, at most 32 steps per call.

Executed action at resulting_step s is the command preceding observation s;
decision at step s is made from observation s. Neither is a success label.
Images remain available through the separate visual-history tool.
"""
    if any(type(n) is not int for n in (start_step, end_step, current_step)):
        raise ValueError("step bounds must be integers")
    if not 0 <= start_step <= end_step <= current_step:
        raise ValueError("range must be within observed history")
    if end_step - start_step + 1 > 32:
        raise ValueError("request at most 32 steps; paginate longer ranges")
    return deepcopy({
        "range": [start_step, end_step],
        "executed_actions": [r for r in ledger
                             if start_step <= r["resulting_step"] <= end_step],
        "decision_history": [r for r in decisions
                             if start_step <= r["step"] <= end_step],
        "memory_is_model_report_not_verified_truth": True,
    })
