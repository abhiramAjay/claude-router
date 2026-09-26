"""Route delegated tasks in the interactive Claude Code CLI, not just the SDK.

`agents.py` does this inside the Agent SDK with a Python hook. Interactive
Claude Code has the same PreToolUse event as a *command* hook: it pipes the
tool call to a process as JSON on stdin and reads a decision back on stdout.
So this is the same decision, taken by a subprocess instead of a coroutine,
and it runs on the CLI's own login -- no ANTHROPIC_API_KEY.

    # .claude/settings.json
    {"hooks": {"PreToolUse": [{"matcher": "Agent|Task", "hooks": [{
        "type": "command", "timeout": 10,
        "command": "PYTHONPATH=\\"$CLAUDE_PROJECT_DIR\\" python3 -m claude_router.cc_hook"}]}]}}

Why `model` and not `subagent_type`
-----------------------------------
The SDK version rewrites `subagent_type` onto five laya-* agents, because an
SDK AgentDefinition is the only place effort can be set. In the CLI that would
also throw away whatever agent Claude picked -- Explore's read-only tools, Plan's
prompt, a project's own agents. The Agent tool takes a `model` override
(haiku / sonnet / opus), so the hook sets that and leaves the agent alone. The
price is the effort lever: the CLI tool has no per-call effort.

Failure is silent by design: sidecar down, bad stdin, anything -- the hook
prints nothing and exits 0, so Claude Code proceeds exactly as it would without
it. Unlike RouterClient's fallback, it does not force Opus: here "no decision"
means the CLI's own default, which is what the user had before installing this.

Cold starts
-----------
After the sidecar has sat idle, its first forward pass takes over 5 s -- longer
than the client waits -- so the first delegation of the day got no decision.
`--warm`, run as a SessionStart hook, sends one throwaway route when a session
opens so the first real one is warm. `/health` would not do: it never touches
the model. The warm-up detaches, so session start is not held up, and waits up
to 60 s itself. It counts as one route in the sidecar's /health stats.

Shared context
--------------
The same PreToolUse call also appends the session's context pool (`pool.py`)
to the brief, after routing, so the router still judges the task and not the
pasted context. `--record`, run on UserPromptSubmit, PostToolUse and
SubagentStop, is what fills the pool, and `--warm` tells the main session the
pool exists, so it does not doubt a subagent that cites earlier results. Injection does not depend on routing: a
dead sidecar still gets the subagent its context.

Every decision is appended to CLAUDE_ROUTER_HOOK_LOG (default
~/Library/Logs/laya-cc-hook.jsonl), so what it chose can be audited afterwards.
"""
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from . import pool
from .client import RouterClient

# `Agent` on current Claude Code, `Task` on older builds -- see agents.py.
DELEGATION_TOOLS = ("Agent", "Task")

# Route.tier -> the Agent tool's `model` enum. Identical today; mapped
# explicitly so a renamed tier fails here instead of as an invalid tool call.
TIER_TO_MODEL = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

LOG_PATH = os.environ.get("CLAUDE_ROUTER_HOOK_LOG",
                          os.path.expanduser("~/Library/Logs/laya-cc-hook.jsonl"))


def decide(event: Dict[str, Any], client: Any) -> Optional[Dict[str, Any]]:
    """The hook's stdout for one PreToolUse event, or None to leave it alone.

    Routes on the brief as Claude wrote it, then appends the session's context
    pool. A router failure costs the model choice, not the context.
    """
    if event.get("tool_name") not in DELEGATION_TOOLS:
        return None
    tool_input = dict(event.get("tool_input") or {})
    original = tool_input.get("prompt") or ""
    task = pool.strip_injected(original) or tool_input.get("description") or ""
    if not task:
        return None

    try:
        route = client.route(task)
    except Exception:
        route = None               # no decision: the CLI keeps its own default
    model = TIER_TO_MODEL.get(route.tier) if route else None

    prompt = pool.inject(original, event.get("session_id")) if original else original
    added = len(prompt) - len(original)
    if model is None and not added:
        return None

    if route is not None:
        _log({"description": tool_input.get("description", ""),
              "subagent_type": tool_input.get("subagent_type"),
              "requested_model": tool_input.get("model"), "chosen_model": model,
              "context_chars": added, **route.to_dict()})

    note = "laya: %s -> %s (%s)" % (tool_input.get("description") or "task",
                                    model or "default model",
                                    route.reason if route else "router unavailable")
    if added:
        note += " +%.1fk chars shared context" % (added / 1000)
    changed = added > 0 or (model is not None and tool_input.get("model") != model)
    if not changed:
        return {"systemMessage": note}
    if model is not None:
        tool_input["model"] = model    # injected when absent, overridden when Claude chose
    tool_input["prompt"] = prompt
    # No permissionDecision: the call still goes through the user's normal
    # permission rules; only its input changes.
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": tool_input},
            "systemMessage": note}


def _log(record: Dict[str, Any]) -> None:
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"ts": time.time(), **record}, default=str) + "\n")
    except OSError:
        pass


WARM_PROMPT = "find where the config file is loaded"
WARM_TIMEOUT = 60.0


def warm(client: Any) -> None:
    """One real forward pass, result discarded, to page the model back in."""
    client.route(WARM_PROMPT)


def main() -> int:
    if sys.argv[1:] == ["--warm"]:
        # SessionStart output becomes context for the main session, which
        # otherwise has no idea its subagents are being handed the pool.
        if pool.ENABLED:
            json.dump({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                              "additionalContext": pool.MAIN_SESSION_NOTE}},
                      sys.stdout)
        # Detach and return at once: SessionStart waits for the hook to exit.
        # The child inherits PYTHONPATH, which is what makes `-m` resolve.
        try:
            subprocess.Popen([sys.executable, "-m", "claude_router.cc_hook", "--warm-now"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            pass
        return 0
    if sys.argv[1:] == ["--warm-now"]:
        pool.prune()
        try:
            warm(RouterClient(timeout=WARM_TIMEOUT, on_error="raise"))
        except Exception:
            pass
        return 0
    if sys.argv[1:] == ["--record"]:
        # Never print: stdout from UserPromptSubmit is added to Claude's context.
        try:
            pool.record(json.load(sys.stdin))
        except Exception:
            pass
        return 0
    try:
        event = json.load(sys.stdin)
        # raise, not fallback: a dead sidecar must mean "no decision", not "Opus".
        out = decide(event, RouterClient(on_error="raise"))
    except Exception:
        return 0
    if out is not None:
        json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
