"""A shared context pool for one Claude Code session, fed and read by hooks.

A subagent starts on an empty context: it sees its brief and nothing else, so a
task that depends on what the user said earlier, or on what another agent found,
is only as good as the brief the orchestrator happened to write. The pool
records both, and the routing hook appends it to every new brief.

What goes in, per session (`~/.claude/laya-pool/<session_id>.jsonl`):

- every user message        UserPromptSubmit carries `prompt`
- every delegated brief     PostToolUse on Agent carries `tool_input`
- every subagent's result   PostToolUse carries `tool_response.content` for a
                            foreground agent; a background one returns before
                            it finishes, so SubagentStop's
                            `last_assistant_message` covers that case

Both result sources fire for a foreground agent, and in this repo the project
and global hooks both fire, so every event carries a key and reads deduplicate.

Why the whole pool and not retrieval
------------------------------------
Within one session the pool is small -- the user's messages plus a few reports,
a few thousand tokens -- and at that size injecting all of it beats choosing:
nothing is missed by a bad match, and on Haiku it costs a fraction of a cent.
Past MAX_TOTAL the oldest agent results are dropped first, and the brief says
how many. Retrieval (laya embeddings or keyword) is for when that starts to
happen routinely; it is not built yet.

The brief PostToolUse reports is the one *after* injection, so everything from
MARKER on is cut before recording -- otherwise each brief would carry the whole
pool and the pool would grow with every task.

Set CLAUDE_ROUTER_POOL=0 to turn off both recording and injection.
"""
import fcntl
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

POOL_DIR = os.environ.get("CLAUDE_ROUTER_POOL_DIR", os.path.expanduser("~/.claude/laya-pool"))
ENABLED = os.environ.get("CLAUDE_ROUTER_POOL", "1") != "0"

MARKER = "<shared-session-context>"
END_MARKER = "</shared-session-context>"
MAX_USER = 1500        # chars per user message
MAX_BRIEF = 400        # chars of each earlier brief -- the result is what matters
MAX_RESULT = 2500      # chars per subagent result
MAX_TOTAL = 12000      # ~3k tokens for the whole section
KEEP_DAYS = 7

DELEGATION_TOOLS = ("Agent", "Task")

_HEADER = (
    "Shared context from this session, recorded automatically. It is background: "
    "your task above takes precedence, and files may have changed since these "
    "results were written, so verify anything you act on."
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:12]


def _path(session_id: Optional[str]) -> Optional[str]:
    # The id becomes a filename, so accept only what Claude Code actually issues.
    if not session_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        return None
    return os.path.join(POOL_DIR, session_id + ".jsonl")


def strip_injected(prompt: str) -> str:
    i = prompt.find(MARKER)
    return prompt if i < 0 else prompt[:i].rstrip()


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...truncated]"


def events_from_hook(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The pool entries one hook payload contributes. Pure, for testing."""
    name = event.get("hook_event_name")
    if name == "UserPromptSubmit":
        prompt = (event.get("prompt") or "").strip()
        if not prompt:
            return []
        return [{"kind": "user", "key": "user:%s" % (event.get("prompt_id") or _digest(prompt)),
                 "text": prompt}]

    if name == "PostToolUse" and event.get("tool_name") in DELEGATION_TOOLS:
        tool_input = event.get("tool_input") or {}
        resp = event.get("tool_response")
        resp = resp if isinstance(resp, dict) else {}
        brief = strip_injected(tool_input.get("prompt") or "")
        agent_id = resp.get("agentId") or _digest(brief)
        out = [{"kind": "brief", "key": "brief:%s" % agent_id, "agent_id": agent_id,
                "agent_type": resp.get("agentType") or tool_input.get("subagent_type"),
                "description": tool_input.get("description", ""), "text": brief}]
        text = "".join(c.get("text", "") for c in resp.get("content") or []
                       if isinstance(c, dict) and c.get("type") == "text")
        if resp.get("status") == "completed" and text.strip():
            out.append(_result(agent_id, resp.get("agentType"), text))
        return out

    if name == "SubagentStop":
        text = event.get("last_assistant_message") or ""
        agent_id = event.get("agent_id")
        if agent_id and text.strip():
            return [_result(agent_id, event.get("agent_type"), text)]
    return []


def _result(agent_id: str, agent_type: Optional[str], text: str) -> Dict[str, Any]:
    return {"kind": "result", "key": "result:%s:%s" % (agent_id, _digest(text)),
            "agent_id": agent_id, "agent_type": agent_type, "text": text.strip()}


def record(event: Dict[str, Any]) -> int:
    """Append this payload's entries to its session's pool. Returns how many."""
    path = _path(event.get("session_id"))
    entries = events_from_hook(event) if ENABLED and path else []
    if not entries:
        return 0
    os.makedirs(POOL_DIR, exist_ok=True)
    lines = "".join(json.dumps({"ts": time.time(), **e}) + "\n" for e in entries)
    # Parallel subagents finish together; the lock keeps their lines whole.
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(lines)
    return len(entries)


def load(session_id: Optional[str]) -> List[Dict[str, Any]]:
    path = _path(session_id)
    if not path or not os.path.exists(path):
        return []
    seen, out = set(), []
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue          # a torn line must not take the pool down with it
            if e.get("key") in seen:
                continue
            seen.add(e.get("key"))
            out.append(e)
    return out


def render(session_id: Optional[str], max_total: Optional[int] = None) -> str:
    """The context section for a new brief, or "" if the pool is empty."""
    max_total = MAX_TOTAL if max_total is None else max_total
    entries = load(session_id)
    users = [_clip(e["text"], MAX_USER) for e in entries if e["kind"] == "user"]

    agents: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for e in entries:
        if e["kind"] not in ("brief", "result"):
            continue
        a = agents.setdefault(e["agent_id"], {"type": None, "description": "",
                                              "brief": "", "results": []})
        a["type"] = a["type"] or e.get("agent_type")
        if e["kind"] == "brief":
            a["description"], a["brief"] = e.get("description", ""), e["text"]
        else:
            a["results"].append(e["text"])

    blocks = []
    for a in agents.values():
        if not a["results"]:
            continue              # still running, or never finished: nothing to share yet
        title = "[%s] %s" % (a["type"] or "agent", a["description"] or "task")
        body = ["### " + title]
        if a["brief"]:
            body.append("Asked: " + _clip(a["brief"], MAX_BRIEF))
        body.extend("Found: " + _clip(r, MAX_RESULT) for r in a["results"])
        blocks.append("\n".join(body))

    if not users and not blocks:
        return ""

    user_part = ("## What the user has asked, oldest first\n" +
                 "\n".join("- " + u.replace("\n", "\n  ") for u in users)) if users else ""
    # User messages always go in: they carry the constraints. Agent results
    # fill what is left, newest first, and the oldest are the ones dropped.
    budget = max_total - len(user_part)
    kept: List[str] = []
    for b in reversed(blocks):
        if len(b) > budget:
            break
        kept.insert(0, b)
        budget -= len(b)
    dropped = len(blocks) - len(kept)

    parts = [_HEADER]
    if user_part:
        parts.append(user_part)
    if blocks:
        head = "## What earlier subagents found, oldest first"
        if dropped:
            head += " (%d older result%s omitted for length)" % (dropped, "" if dropped == 1 else "s")
        parts.append("\n\n".join([head] + kept))
    return "\n\n%s\n%s\n%s" % (MARKER, "\n\n".join(parts), END_MARKER)


def inject(prompt: str, session_id: Optional[str]) -> str:
    """`prompt` with the pool appended -- once, however many hooks run."""
    if not ENABLED or MARKER in prompt:
        return prompt
    return prompt + render(session_id)


def prune(now: Optional[float] = None) -> int:
    """Delete pools untouched for KEEP_DAYS. Returns how many."""
    cutoff = (now or time.time()) - KEEP_DAYS * 86400
    removed = 0
    try:
        names = os.listdir(POOL_DIR)
    except OSError:
        return 0
    for name in names:
        p = os.path.join(POOL_DIR, name)
        try:
            if name.endswith(".jsonl") and os.path.getmtime(p) < cutoff:
                os.remove(p)
                removed += 1
        except OSError:
            pass
    return removed
