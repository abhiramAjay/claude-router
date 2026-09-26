"""Shared context pool tests. Payloads mirror what Claude Code 2.1.283 actually
sends, captured from a probe project -- field names included."""
import json
import os
import subprocess
import sys

import pytest

from claude_router import cc_hook, pool
from claude_router.router import Route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SID = "60222ff2-3ee0-4d62-a692-2fbea16e437b"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, "POOL_DIR", str(tmp_path / "pool"))
    monkeypatch.setattr(pool, "ENABLED", True)
    monkeypatch.setattr(cc_hook, "LOG_PATH", str(tmp_path / "hook.jsonl"))
    return tmp_path


def user(text, pid="p1"):
    return {"session_id": SID, "hook_event_name": "UserPromptSubmit", "prompt_id": pid,
            "prompt": text}


def post(prompt, result, agent_id="a1", desc="Find it", agent_type="Explore",
         status="completed"):
    content = [{"type": "text", "text": result}] if result else []
    return {"session_id": SID, "hook_event_name": "PostToolUse", "tool_name": "Agent",
            "tool_input": {"description": desc, "prompt": prompt, "subagent_type": agent_type,
                           "model": "haiku"},
            "tool_response": {"status": status, "agentId": agent_id, "agentType": agent_type,
                              "content": content}}


def stop(result, agent_id="a1", agent_type="Explore"):
    return {"session_id": SID, "hook_event_name": "SubagentStop", "agent_id": agent_id,
            "agent_type": agent_type, "last_assistant_message": result}


def test_empty_pool_injects_nothing():
    assert pool.render(SID) == ""
    assert pool.inject("do X", SID) == "do X"


def test_records_user_brief_and_result():
    pool.record(user("rename field foo to bar everywhere; keep the API backwards compatible"))
    pool.record(post("find every use of foo", "foo is used in api.py:10 and worker.py:22"))
    ctx = pool.render(SID)
    assert "keep the API backwards compatible" in ctx
    assert "[Explore] Find it" in ctx
    assert "Asked: find every use of foo" in ctx
    assert "api.py:10 and worker.py:22" in ctx
    assert ctx.strip().startswith(pool.MARKER) and ctx.strip().endswith(pool.END_MARKER)


def test_foreground_agent_result_is_not_duplicated():
    # A foreground agent fires SubagentStop *and* PostToolUse with the same text.
    pool.record(stop("found it at a.py:1"))
    pool.record(post("find it", "found it at a.py:1"))
    assert pool.render(SID).count("found it at a.py:1") == 1


def test_duplicate_hooks_are_deduplicated():
    # Project and global hooks both fire in this repo.
    for _ in range(2):
        pool.record(user("hello", pid="same"))
        pool.record(post("find it", "result"))
    ctx = pool.render(SID)
    assert ctx.count("- hello") == 1 and ctx.count("Found: result") == 1


def test_background_agent_result_arrives_via_subagent_stop():
    pool.record(post("long investigation", None, agent_id="bg", status="async_launched"))
    assert "long investigation" not in pool.render(SID)      # nothing to share while running
    pool.record(stop("the investigation found Z", agent_id="bg"))
    ctx = pool.render(SID)
    assert "Asked: long investigation" in ctx and "found Z" in ctx


def test_injected_context_is_stripped_before_recording():
    pool.record(post("first task", "first result", agent_id="a1"))
    brief = pool.inject("second task", SID)
    assert pool.MARKER in brief
    pool.record(post(brief, "second result", agent_id="a2"))
    ctx = pool.render(SID)
    assert "Asked: second task" in ctx
    assert ctx.count(pool.MARKER) == 1                         # no pool inside the pool


def test_inject_is_idempotent():
    pool.record(user("context"))
    once = pool.inject("task", SID)
    assert pool.inject(once, SID) == once


def test_budget_drops_oldest_results_and_says_so():
    pool.record(user("the constraint"))
    for i in range(6):
        pool.record(post("task %d" % i, ("result %d " % i) + "x" * 900, agent_id="a%d" % i))
    ctx = pool.render(SID, max_total=2500)
    assert "the constraint" in ctx                             # user messages always kept
    assert "result 5" in ctx and "result 0" not in ctx         # newest kept, oldest dropped
    assert "older results omitted" in ctx


def test_long_entries_are_clipped():
    pool.record(post("t", "y" * 10000))
    ctx = pool.render(SID)
    assert "[...truncated]" in ctx and len(ctx) < 4000


def test_sessions_are_separate():
    pool.record(user("session one"))
    assert pool.render("other-session") == ""


def test_hostile_session_ids_are_refused():
    assert pool.record(dict(user("x"), session_id="../../etc/passwd")) == 0
    assert pool.record(dict(user("x"), session_id=None)) == 0


def test_torn_lines_are_skipped(isolated):
    pool.record(user("good"))
    with open(pool._path(SID), "a") as f:
        f.write('{"kind": "user", "te')
    assert "good" in pool.render(SID)


def test_disabled_records_and_injects_nothing(monkeypatch):
    monkeypatch.setattr(pool, "ENABLED", False)
    assert pool.record(user("x")) == 0
    assert pool.inject("task", SID) == "task"


def test_prune_removes_only_stale_pools(isolated):
    pool.record(user("fresh"))
    old = os.path.join(pool.POOL_DIR, "old.jsonl")
    open(old, "w").close()
    os.utime(old, (0, 0))
    assert pool.prune() == 1
    assert not os.path.exists(old) and os.path.exists(pool._path(SID))


# -- through the routing hook --------------------------------------------------

class Stub:
    def __init__(self, tier="haiku", exc=None):
        self.tier, self.exc, self.prompts = tier, exc, []

    def route(self, prompt):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return Route(tier=self.tier, model="claude-" + self.tier, effort=None,
                     difficulty=1.0, domain="code", reason="stub")


def pre(prompt):
    return {"session_id": SID, "hook_event_name": "PreToolUse", "tool_name": "Agent",
            "tool_input": {"description": "d", "prompt": prompt}}


def test_routes_on_the_task_then_appends_context():
    pool.record(user("x" * 3000))                              # would inflate a length-biased router
    client = Stub("haiku")
    out = cc_hook.decide(pre("find the config loader"), client)
    assert client.prompts == ["find the config loader"]        # routed on the bare task
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["model"] == "haiku"
    assert updated["prompt"].startswith("find the config loader") and pool.MARKER in updated["prompt"]
    assert "shared context" in out["systemMessage"]


def test_dead_router_still_gets_context():
    pool.record(user("the constraint"))
    out = cc_hook.decide(pre("task"), Stub(exc=OSError("down")))
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert "model" not in updated                              # CLI default, not Opus
    assert "the constraint" in updated["prompt"]


def test_second_hook_routes_the_bare_task_and_does_not_reinject():
    pool.record(user("ctx"))
    first = cc_hook.decide(pre("task"), Stub())["hookSpecificOutput"]["updatedInput"]["prompt"]
    client = Stub()
    second = cc_hook.decide(pre(first), client)
    assert client.prompts == ["task"]
    assert "hookSpecificOutput" not in second or \
        second["hookSpecificOutput"]["updatedInput"]["prompt"] == first


def test_record_subprocess_prints_nothing(isolated):
    # UserPromptSubmit stdout becomes context -- the recorder must stay silent.
    env = dict(os.environ, PYTHONPATH=ROOT, CLAUDE_ROUTER_POOL_DIR=str(isolated / "pool"),
               CLAUDE_ROUTER_POOL="1")
    for payload in (json.dumps(user("hi")), "not json", ""):
        r = subprocess.run([sys.executable, "-m", "claude_router.cc_hook", "--record"],
                           input=payload, capture_output=True, text=True, env=env, timeout=30)
        assert r.returncode == 0 and r.stdout == ""
    assert "hi" in open(os.path.join(isolated, "pool", SID + ".jsonl")).read()


@pytest.mark.parametrize("enabled", ["1", "0"])
def test_session_start_tells_the_main_session_about_the_pool(enabled, isolated):
    env = dict(os.environ, PYTHONPATH=ROOT, CLAUDE_ROUTER_POOL=enabled,
               CLAUDE_ROUTER_URL="http://127.0.0.1:9", CLAUDE_ROUTER_POOL_DIR=str(isolated / "p"))
    r = subprocess.run([sys.executable, "-m", "claude_router.cc_hook", "--warm"],
                       input='{"hook_event_name": "SessionStart", "source": "startup"}',
                       capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0
    if enabled == "0":
        assert r.stdout == ""            # no pool, so nothing to explain
        return
    out = json.loads(r.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    assert pool.MARKER in out["additionalContext"]    # names the section it will see
