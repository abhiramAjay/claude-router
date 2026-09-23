"""Route each *delegated* task to a Claude model, inside the Claude Agent SDK.

The Agent SDK is Claude Code as a library, so there is nothing to clone -- the
only piece worth owning is the decision about which model does which task. This
module wires `RouterClient` into that decision at the one point where it is free.

Why subagents and not the main loop
-----------------------------------
`ClaudeSDKClient.set_model()` exists, but prompt caches are model-scoped and a
coding thread resends its whole history every turn, so switching mid-session
pays full price for that history instead of the ~0.1x cache-read rate (measured
in this repo's README: $0.0325 -> $0.0775 on a 10k history, and it grows). A
subagent starts on a fresh context, so there is no cache to forfeit -- picking
its model costs nothing and saves the whole task.

How the decision is taken away from Claude
------------------------------------------
`AgentDefinition.model`/`.effort` are static, and the `SubagentStart` hook can
only append context -- it cannot change the model. So the routing happens one
step earlier: a `PreToolUse` hook on the delegation tool reads the delegated
prompt, asks laya (~75 ms, local, no tokens), and sets `subagent_type` via
`updatedInput` before the subagent is spawned. Claude proposes the task; laya
picks who runs it.

That tool is named `Agent` on current Claude Code and `Task` on older builds, so
`DELEGATION_TOOLS` matches both -- and the model frequently omits
`subagent_type` altogether (it defaults to general-purpose), so the hook injects
the key rather than assuming it is there to rewrite.

    from claude_router.agents import routing_options

    options = routing_options(cwd=".")
    async for msg in query(prompt="...", options=options):
        ...
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, HookMatcher

from .client import RouterClient
from .router import Route

# The router's output space is (tier, effort), and AgentDefinition is static, so
# there is exactly one agent per reachable combination -- five, not three. That
# keeps both of the router's levers instead of collapsing effort into the tier.
_TIER_AGENTS = [
    ("haiku",  None),
    ("sonnet", "medium"),
    ("sonnet", "high"),
    ("opus",   "high"),
    ("opus",   "xhigh"),
]

# The delegation tool. `Agent` on Claude Code 2.1.x; `Task` on older builds --
# matched by name rather than pinned, because getting this wrong is silent: the
# hook simply never fires and every subagent runs on the default model.
DELEGATION_TOOLS = ("Agent", "Task")

# The neutral name the main agent is told to delegate to. It is itself a real
# agent (mid-tier) so the call still validates if the hook is absent or fails.
DEFAULT_AGENT = "laya-sonnet-high"

_PROMPT = (
    "You are a delegated worker. Complete exactly the task you were given and "
    "report the result. Do not expand the scope."
)


def agent_name(tier: str, effort: Optional[str]) -> str:
    return "laya-%s" % tier if effort is None else "laya-%s-%s" % (tier, effort)


def agent_name_for(route: Route) -> str:
    """The agent implementing `route`, or the default if the pair is unreachable.

    A fallback Route (sidecar down) carries model `claude-opus-5` with effort
    `high`, which is a real tier/effort pair, so it lands on the Opus agent --
    the same degrade-to-the-good-model behaviour the HTTP client already has.
    """
    name = agent_name(route.tier, route.effort)
    return name if name in TIER_AGENTS else DEFAULT_AGENT


TIER_AGENTS: Dict[str, AgentDefinition] = {
    agent_name(tier, effort): AgentDefinition(
        description="Delegated work at the %s tier%s. Selected by the router, not by you."
                    % (tier, "" if effort is None else " (%s effort)" % effort),
        prompt=_PROMPT,
        model=tier,
        effort=effort,
    )
    for tier, effort in _TIER_AGENTS
}

# Appended to the system prompt: the main agent picks *what* to delegate, the
# hook picks *who* to. Naming one agent keeps its choice out of the cost model.
DELEGATION_INSTRUCTION = (
    "When you delegate with the Task tool, always pass subagent_type=%r. "
    "The model that runs it is chosen for you by a local classifier; do not "
    "try to pick a tier yourself." % DEFAULT_AGENT
)


@dataclass
class TaskRouter:
    """PreToolUse hook that re-points each Task call at a laya-chosen agent."""

    # mode=None defers to the sidecar's default, which is the trained head.
    # Measured on 40 real delegated briefs: head 38/40, difficulty score 20/40.
    client: RouterClient = field(default_factory=RouterClient)
    log: List[Dict[str, Any]] = field(default_factory=list)

    async def __call__(self, input: Dict[str, Any], tool_use_id: Optional[str],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        if input.get("tool_name") not in DELEGATION_TOOLS:
            return {}
        tool_input = dict(input.get("tool_input") or {})
        prompt = tool_input.get("prompt") or tool_input.get("description") or ""
        if not prompt:
            return {}

        route = self.client.route(prompt)
        chosen = agent_name_for(route)
        self.log.append({"description": tool_input.get("description", ""),
                         "requested": tool_input.get("subagent_type"),
                         "chosen": chosen, **route.to_dict()})

        if chosen == tool_input.get("subagent_type"):
            return {}
        tool_input["subagent_type"] = chosen      # injected when absent, not just rewritten
        return {
            "hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": tool_input},
            "systemMessage": "laya: %s (%s)" % (chosen, route.reason),
        }

    def summary(self) -> str:
        if not self.log:
            return "no tasks delegated"
        rows = ["%-8s %-22s %s" % (r["tier"], r["chosen"], r["reason"]) for r in self.log]
        return "\n".join(rows)


def routing_options(router: Optional[TaskRouter] = None, **kwargs: Any) -> ClaudeAgentOptions:
    """`ClaudeAgentOptions` with the tier agents registered and the hook attached.

    Any `agents` or `hooks` passed in are merged, not replaced, so this composes
    with a project's own agents. `system_prompt` is left alone -- append
    `DELEGATION_INSTRUCTION` to yours if you set one.
    """
    router = router or TaskRouter()
    agents = dict(TIER_AGENTS, **(kwargs.pop("agents", None) or {}))
    hooks = dict(kwargs.pop("hooks", None) or {})
    matcher = HookMatcher(matcher="|".join(DELEGATION_TOOLS), hooks=[router])
    hooks["PreToolUse"] = [matcher] + hooks.get("PreToolUse", [])
    return ClaudeAgentOptions(agents=agents, hooks=hooks, **kwargs)
