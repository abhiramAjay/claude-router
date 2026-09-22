"""Spike: route delegated tasks to Claude models inside the Claude Agent SDK.

    python spike_subagents.py --dry-run      # routing only, no API call, no tokens
    python spike_subagents.py                # full run: needs credentials

`--dry-run` feeds the classifier the kind of prompts a main agent actually emits
when it delegates, and prints the tier each one lands on. That exercises the
sidecar and the tier table without spending anything; the full run puts the same
decision behind a real Task tool call.
"""
import argparse
import asyncio
import sys

from claude_router.agents import (DEFAULT_AGENT, DELEGATION_INSTRUCTION, DELEGATION_TOOLS,
                                  TaskRouter, agent_name_for, routing_options)

# Delegated-task prompts, roughly ordered by the work they imply.
SAMPLE_TASKS = [
    "List the files in the claude_router package.",
    "Find every call site of RouterClient.route in this repo.",
    "Rename the variable `diff` to `difficulty` in router.py.",
    "Add pagination to the /route endpoint, with tests.",
    "Summarize what server.py does in two sentences.",
    "Design a distributed rate limiter for the sidecar that survives restarts.",
    "Drop the users table in prod.",
]


def dry_run() -> int:
    router = TaskRouter()
    if not router.client.health().get("ok"):
        print("sidecar unreachable -- every task will degrade to Opus.", file=sys.stderr)
    print("%-8s %-22s %-5s %s" % ("TIER", "AGENT", "DIFF", "TASK"))
    for task in SAMPLE_TASKS:
        route = router.client.route(task)
        print("%-8s %-22s %-5.2f %s" % (route.tier, agent_name_for(route), route.difficulty,
                                        task[:52]))
    return 0


async def live_run(prompt: str, model: str) -> int:
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

    router = TaskRouter()
    options = routing_options(
        router=router,
        system_prompt={"type": "preset", "preset": "claude_code", "append": DELEGATION_INSTRUCTION},
        # Delegation only. Given Read/Glob/Grep the main agent just does the
        # work inline -- measured: 0 delegations, $0.78, all of it on the
        # orchestrator. The hook can only fire on a delegation call, so
        # delegating has to be the one way forward, not a polite suggestion.
        allowed_tools=list(DELEGATION_TOOLS),
        permission_mode="acceptEdits",
        max_turns=12,
        model=model,
    )
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            print("\n\n--- routed %d task(s) ---" % len(router.log))
            print(router.summary())
            # The hook firing only proves it fired. `model_usage` is keyed by
            # the model that actually served the tokens, so a Haiku entry is
            # the proof the rewrite reached the subagent.
            for model, usage in sorted((message.model_usage or {}).items()):
                print("%-22s %7d in %7d out  $%.4f"
                      % (model, usage["inputTokens"], usage["outputTokens"], usage["costUSD"]))
            if message.total_cost_usd:
                print("total: $%.4f" % message.total_cost_usd)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spike_subagents")
    ap.add_argument("--model", default="sonnet", help="model for the orchestrator itself")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the routing table only; no API call")
    ap.add_argument("prompt", nargs="?",
                    # Three tasks that should land on three different tiers.
                    default="Delegate each of these to its own subagent, in parallel, "
                            "and report their answers verbatim: "
                            "(1) list the .py files in claude_router/; "
                            "(2) name the two tunables that set the tier cuts in "
                            "router.py; "
                            "(3) design a scheme for the sidecar to survive restarts "
                            "without losing its route statistics -- cover format, "
                            "flush cadence, crash safety and what loss is acceptable.")
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run()
    print("delegating via %s; laya re-points each Task.\n" % DEFAULT_AGENT, file=sys.stderr)
    return asyncio.run(live_run(args.prompt, args.model))


if __name__ == "__main__":
    raise SystemExit(main())
