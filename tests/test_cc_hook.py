"""Claude Code command-hook tests. No network, no torch: `decide` is driven with
a stub router, and `main` is run as the real subprocess Claude Code would spawn."""
import json
import os
import subprocess
import sys
import time

import pytest

from claude_router import cc_hook
from claude_router.router import Route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StubClient:
    def __init__(self, tier="haiku", exc=None):
        self.tier, self.exc, self.prompts = tier, exc, []

    def route(self, prompt):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return Route(tier=self.tier, model="claude-%s" % self.tier, effort=None,
                     difficulty=1.0, domain="code", reason="stub")


@pytest.fixture(autouse=True)
def log_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_hook, "LOG_PATH", str(tmp_path / "hook.jsonl"))
    monkeypatch.setattr(cc_hook.pool, "POOL_DIR", str(tmp_path / "pool"))
    return tmp_path / "hook.jsonl"


def event(tool_name="Agent", **tool_input):
    return {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": tool_input}


def test_sets_model_and_keeps_the_agent():
    out = cc_hook.decide(event(prompt="find where X is defined", description="find X",
                               subagent_type="Explore"), StubClient("haiku"))
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated == {"prompt": "find where X is defined", "description": "find X",
                       "subagent_type": "Explore", "model": "haiku"}
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in out["hookSpecificOutput"]


def test_overrides_a_model_claude_picked():
    out = cc_hook.decide(event(prompt="design a rate limiter", model="haiku"), StubClient("opus"))
    assert out["hookSpecificOutput"]["updatedInput"]["model"] == "opus"


def test_no_rewrite_when_already_matching(log_to_tmp):
    out = cc_hook.decide(event(prompt="p", model="sonnet"), StubClient("sonnet"))
    assert "hookSpecificOutput" not in out
    assert len(log_to_tmp.read_text().splitlines()) == 1     # still recorded


def test_matches_the_old_tool_name():
    assert cc_hook.decide(event("Task", prompt="p"), StubClient()) is not None


def test_ignores_other_tools_and_empty_prompts():
    client = StubClient()
    assert cc_hook.decide(event("Bash", command="ls"), client) is None
    assert cc_hook.decide(event(prompt=""), client) is None
    assert client.prompts == []


def test_unknown_tier_is_left_alone():
    assert cc_hook.decide(event(prompt="p"), StubClient("fable")) is None


def test_warm_runs_a_real_route():
    client = StubClient()
    cc_hook.warm(client)
    assert client.prompts == [cc_hook.WARM_PROMPT]    # a forward pass, not /health


def run_hook(stdin, url, *args):
    env = dict(os.environ, PYTHONPATH=ROOT, CLAUDE_ROUTER_URL=url,
               CLAUDE_ROUTER_HOOK_LOG=os.devnull, CLAUDE_ROUTER_POOL="0")
    return subprocess.run([sys.executable, "-m", "claude_router.cc_hook", *args], input=stdin,
                          capture_output=True, text=True, env=env, timeout=30)


def test_dead_sidecar_means_no_decision_not_opus():
    # Port 9 (discard) is closed on loopback: connection refused, instantly.
    r = run_hook(json.dumps(event(prompt="rename a variable")), "http://127.0.0.1:9")
    assert r.returncode == 0 and r.stdout == ""


def test_garbage_stdin_is_harmless():
    r = run_hook("not json", "http://127.0.0.1:9")
    assert r.returncode == 0 and r.stdout == ""


@pytest.mark.parametrize("flag", ["--warm", "--warm-now"])
def test_warm_is_silent_and_harmless_with_a_dead_sidecar(flag):
    r = run_hook("", "http://127.0.0.1:9", flag)
    assert r.returncode == 0 and r.stdout == ""


# --- the project and global hooks both firing for one event -------------------

def _run(args, payload, env):
    return subprocess.run([sys.executable, "-m", "claude_router.cc_hook", *args],
                          input=json.dumps(payload), capture_output=True, text=True,
                          env=env, timeout=30)


@pytest.fixture
def claim_env(tmp_path):
    return dict(os.environ, PYTHONPATH=ROOT, CLAUDE_ROUTER_CLAIM_DIR=str(tmp_path / "claims"),
                CLAUDE_ROUTER_URL="http://127.0.0.1:9", CLAUDE_ROUTER_POOL="1",
                CLAUDE_ROUTER_POOL_DIR=str(tmp_path / "pool"),
                CLAUDE_ROUTER_HOOK_LOG=str(tmp_path / "hook.jsonl"))


def test_claim_is_taken_once_per_event(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_hook, "CLAIM_DIR", str(tmp_path / "claims"))
    event = {"session_id": "s1", "tool_use_id": "toolu_1", "tool_input": {"prompt": "x"}}
    assert cc_hook.claim(event, "pre") is True
    assert cc_hook.claim(event, "pre") is False                     # the second copy
    assert cc_hook.claim(dict(event, tool_use_id="toolu_2"), "pre") is True
    assert cc_hook.claim(dict(event, session_id="s2"), "pre") is True
    assert cc_hook.claim(event, "session") is True                  # kinds don't collide


def test_claim_falls_back_to_the_input_without_a_tool_use_id(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_hook, "CLAIM_DIR", str(tmp_path / "claims"))
    event = {"session_id": "s1", "tool_input": {"prompt": "find x"}}
    assert cc_hook.claim(event, "pre") is True
    assert cc_hook.claim(event, "pre") is False
    assert cc_hook.claim(dict(event, tool_input={"prompt": "find y"}), "pre") is True


def test_stale_claim_is_retaken(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_hook, "CLAIM_DIR", str(tmp_path / "claims"))
    event = {"session_id": "s1", "source": "startup"}
    assert cc_hook.claim(event, "session") is True
    old = time.time() - cc_hook.CLAIM_WINDOW - 1
    for name in os.listdir(tmp_path / "claims"):
        os.utime(tmp_path / "claims" / name, (old, old))
    assert cc_hook.claim(event, "session") is True                  # a later /clear


def test_claim_without_session_never_suppresses(tmp_path, monkeypatch):
    monkeypatch.setattr(cc_hook, "CLAIM_DIR", str(tmp_path / "claims"))
    assert cc_hook.claim({"source": "startup"}, "session") is True
    assert cc_hook.claim({"source": "startup"}, "session") is True


def test_unwritable_claim_dir_fails_open(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(cc_hook, "CLAIM_DIR", str(blocker / "claims"))
    event = {"session_id": "s1", "tool_use_id": "t"}
    assert cc_hook.claim(event, "pre") is True and cc_hook.claim(event, "pre") is True


def test_duplicate_pretooluse_hook_stays_silent(claim_env):
    event = {"session_id": "s1", "tool_use_id": "toolu_9", "hook_event_name": "PreToolUse",
             "tool_name": "Agent", "tool_input": {"description": "d", "prompt": "find x"}}
    _run(["--record"], {"session_id": "s1", "hook_event_name": "UserPromptSubmit",
                        "prompt": "earlier message"}, claim_env)
    first, second = _run([], event, claim_env), _run([], event, claim_env)
    assert first.returncode == second.returncode == 0
    assert "updatedInput" in first.stdout                           # pool still injected
    assert second.stdout == ""


def test_duplicate_session_start_adds_the_note_once(claim_env):
    event = {"session_id": "s1", "hook_event_name": "SessionStart", "source": "startup"}
    first, second = _run(["--warm"], event, claim_env), _run(["--warm"], event, claim_env)
    assert "additionalContext" in first.stdout
    assert second.stdout == ""
