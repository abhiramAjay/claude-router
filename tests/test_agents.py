"""Subagent routing tests. No API key, no network, no torch -- the hook is
driven with a stub router so the rewrite logic is testable on its own."""
import asyncio

import pytest

from claude_router.agents import (DEFAULT_AGENT, DELEGATION_INSTRUCTION, DELEGATION_TOOLS,
                                  TIER_AGENTS, TaskRouter, agent_name_for, routing_options)
from claude_router.router import Route


class StubClient:
    def __init__(self, route):
        self.route_value, self.prompts = route, []

    def route(self, prompt):
        self.prompts.append(prompt)
        return self.route_value


def make_route(tier="opus", effort="xhigh", **kw):
    return Route(tier=tier, model="claude-%s" % tier, effort=effort,
                 difficulty=kw.pop("difficulty", 2.3), domain="code",
                 reason=kw.pop("reason", "stub"), **kw)


def call(hook, tool_name="Agent", **tool_input):
    return asyncio.run(hook({"tool_name": tool_name, "tool_input": tool_input}, "tu_1", {}))


def test_every_reachable_route_has_an_agent():
    # The router's real output space -- if router.py grows a tier or an effort
    # level, this is what notices.
    for tier, efforts in (("haiku", [None]), ("sonnet", ["medium", "high"]),
                          ("opus", ["high", "xhigh"])):
        for effort in efforts:
            assert agent_name_for(make_route(tier, effort)) in TIER_AGENTS


def test_agents_carry_both_levers():
    assert TIER_AGENTS["laya-haiku"].model == "haiku"
    assert TIER_AGENTS["laya-haiku"].effort is None      # Haiku 4.5 rejects effort
    assert TIER_AGENTS["laya-opus-xhigh"].effort == "xhigh"
    assert DEFAULT_AGENT in TIER_AGENTS
    assert DEFAULT_AGENT in DELEGATION_INSTRUCTION


def test_hook_rewrites_subagent_type():
    hook = TaskRouter(client=StubClient(make_route("haiku", None)))
    out = call(hook, prompt="rename a variable", subagent_type=DEFAULT_AGENT)
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["subagent_type"] == "laya-haiku"
    assert updated["prompt"] == "rename a variable"      # nothing else touched
    assert hook.client.prompts == ["rename a variable"]


def test_hook_is_a_noop_when_the_route_already_matches():
    hook = TaskRouter(client=StubClient(make_route("opus", "xhigh")))
    assert call(hook, prompt="design a rate limiter", subagent_type="laya-opus-xhigh") == {}
    assert len(hook.log) == 1                            # still recorded


def test_hook_ignores_other_tools_without_routing():
    hook = TaskRouter(client=StubClient(make_route()))
    assert call(hook, tool_name="Bash", command="ls") == {}
    assert hook.client.prompts == []                     # no 75ms spent


@pytest.mark.parametrize("tool_name", DELEGATION_TOOLS)
def test_both_delegation_tool_names_are_matched(tool_name):
    # `Agent` on Claude Code 2.1.x, `Task` on older builds. Getting this wrong
    # is silent -- the hook never fires and everything runs on the default model.
    hook = TaskRouter(client=StubClient(make_route("haiku", None)))
    out = call(hook, tool_name=tool_name, prompt="rename a variable")
    assert out["hookSpecificOutput"]["updatedInput"]["subagent_type"] == "laya-haiku"


def test_subagent_type_is_injected_when_the_model_omits_it():
    # Measured against Claude Code 2.1.x: the model routinely calls Agent with
    # only `description` and `prompt`, letting the type default. Rewrite-only
    # logic would leave those on the default model.
    hook = TaskRouter(client=StubClient(make_route("opus", "xhigh")))
    out = call(hook, prompt="design a rate limiter", description="design")
    assert out["hookSpecificOutput"]["updatedInput"]["subagent_type"] == "laya-opus-xhigh"


def test_hook_falls_back_to_description_then_gives_up():
    hook = TaskRouter(client=StubClient(make_route("sonnet", "medium")))
    out = call(hook, description="add pagination", subagent_type=DEFAULT_AGENT)
    assert out["hookSpecificOutput"]["updatedInput"]["subagent_type"] == "laya-sonnet-medium"
    assert call(hook, subagent_type=DEFAULT_AGENT) == {}
    assert len(hook.log) == 1


def test_unreachable_pair_lands_on_the_default():
    assert agent_name_for(make_route("sonnet", "xhigh")) == DEFAULT_AGENT


def test_sidecar_down_degrades_to_opus_not_to_a_crash():
    from claude_router.client import RouterClient
    down = RouterClient(url="http://127.0.0.1:1")         # nothing listens here
    route = down.route("anything")
    assert "unavailable" in route.reason
    assert agent_name_for(route) == "laya-opus-high"


def test_routing_options_merges_rather_than_replaces():
    from claude_agent_sdk import AgentDefinition, HookMatcher
    mine = AgentDefinition(description="d", prompt="p")
    other = HookMatcher(matcher="Bash", hooks=[])
    opts = routing_options(agents={"mine": mine}, hooks={"PreToolUse": [other]},
                           max_turns=3)
    assert opts.agents["mine"] is mine
    assert set(TIER_AGENTS) <= set(opts.agents)
    assert opts.hooks["PreToolUse"][0].matcher == "Agent|Task"   # ours runs first
    assert other in opts.hooks["PreToolUse"]
    assert opts.max_turns == 3
