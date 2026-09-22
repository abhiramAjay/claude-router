"""Calibrate the router on *delegated* task phrasing, not user phrasing.

`tests/test_routing.py` labels prompts a person types. That is not what the
router sees inside the Agent SDK: the orchestrator rewrites each task before
delegating it, adding repo context, file paths and acceptance criteria, so the
same work arrives as a several-hundred-character brief. Cuts tuned on one
distribution do not transfer to the other -- measuring that is this file's job.

    python eval_delegated.py                 # score the saved corpus (free, no API)
    python eval_delegated.py --capture       # regenerate it (costs money, ~$0.15/case)

`--capture` runs a real orchestrator over each user request and records the
prompt it hands to the subagent. The subagent is redirected to a no-op Haiku
agent, so the paraphrase is paid for and the work is not. The corpus is cached
in `delegated_corpus.json` so scoring and re-tuning stay free.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delegated_corpus.json")

# User requests, as typed to Claude Code, with the tier the delegated task is
# worth. Deliberately *not* the cases in tests/test_routing.py: chitchat and
# trivia never get delegated, so the delegated distribution is truncated at the
# bottom -- the cheap tier's job here is mechanical repo lookups, not greetings.
# User requests, as typed to Claude Code, with the tier the delegated task is
# worth. Deliberately *not* the cases in tests/test_routing.py: chitchat and
# trivia never get delegated, so the delegated distribution is truncated at the
# bottom -- the cheap tier's job here is mechanical repo lookups, not greetings.
#
# The labelling rule, applied consistently:
#   haiku  -- retrieve something that already exists; no judgement
#   sonnet -- a contained change, or an explanation with one right answer
#   opus   -- design, trade-offs, proof, or a judgement call across the codebase
#
# Weighted toward the sonnet/opus boundary, which is where the classifier
# actually fails; there is no point adding more cases where it already wins.
CASES = [
    # ---- lookup: one tool call and a verbatim answer ----
    ("Find out which port the sidecar listens on by default.", "haiku"),
    ("List the .py files in the claude_router package.", "haiku"),
    ("Tell me the name of the regex constant that handles high-stakes escalation.", "haiku"),
    ("Find every place RouterClient.route is called in this repo.", "haiku"),
    ("Read the docstring of messages_kwargs and tell me what it says.", "haiku"),
    ("What version of laya does the server extra require?", "haiku"),
    ("What model does the client fall back to when the sidecar is down?", "haiku"),
    ("List the environment variables the client reads.", "haiku"),
    ("What is the launchd label for the daemon?", "haiku"),
    ("How many labelled cases are in tests/test_routing.py?", "haiku"),
    ("What are the three tier names in the TIERS table?", "haiku"),
    ("Find where the default HTTP timeout is set in client.py.", "haiku"),
    # ---- contained change, or an explanation with one right answer ----
    ("Explain how the high-stakes guard interacts with the difficulty cuts.", "sonnet"),
    ("Add a --timeout flag to laya-routerd and wire it through to the server.", "sonnet"),
    ("Why would route() return NaN difficulty? Track down the cause.", "sonnet"),
    ("Write tests covering the case where the sidecar returns a 500.", "sonnet"),
    ("Summarize what server.py does and where its error handling is weak.", "sonnet"),
    ("Add a /stats endpoint that returns the by_tier counts as CSV.", "sonnet"),
    ("Make /health return a JSON error body listing the allowed paths on a 404.", "sonnet"),
    ("Explain why messages_kwargs omits the thinking parameter for Haiku.", "sonnet"),
    ("Log every routing decision to a file, one JSON object per line.", "sonnet"),
    ("Write a test that the high-stakes regex matches 'force-push' but not "
     "'force pushing a commit is fine'.", "sonnet"),
    ("Work out why the daemon takes 98s to start under launchd but 20s in the foreground.", "sonnet"),
    ("Refactor ClaudeRouter.route so tier selection and effort selection are separate.", "sonnet"),
    ("Add a --json flag to the eval script that prints its results as JSON.", "sonnet"),
    ("Explain the difference between RouterClient and ClaudeRouter and when to use each.", "sonnet"),
    ("Make the client retry once on a timeout before falling back to Opus.", "sonnet"),
    ("Document the CHEAP_DOMAINS constant with an example of when it changes a decision.", "sonnet"),
    # ---- design, trade-offs, or a judgement call ----
    ("Design a scheme for the sidecar to survive restarts without losing its "
     "route statistics. Cover format, flush cadence and crash safety.", "opus"),
    ("Rework the router so cuts can be A/B tested per project without "
     "restarting the daemon, and argue for the design you pick.", "opus"),
    ("Work out whether this cascade actually saves money versus Opus at low "
     "effort, and show the arithmetic.", "opus"),
    ("Redesign the sidecar to serve several models' routing tables at once "
     "without reloading the checkpoint.", "opus"),
    ("Design a way to measure whether routing saves money in production, and "
     "argue for the metric you choose over the alternatives.", "opus"),
    ("The difficulty score correlates with prompt length. Design a fix and "
     "justify it against the other options.", "opus"),
    ("Decide whether this project should keep the HTTP sidecar or move to a "
     "Unix socket, and make the case either way.", "opus"),
    ("Design a cache so repeated prompts skip the classifier, and reason about "
     "what happens to correctness when the cuts change.", "opus"),
    ("Propose an architecture for routing across several providers rather than "
     "just Anthropic, and defend the abstraction you choose.", "opus"),
    ("Evaluate whether laya is the right classifier for this job at all and "
     "recommend what to do about it.", "opus"),
    ("Design a rollout plan for changing the tier cuts without regressing "
     "anyone's costs, and say how you would know if it went wrong.", "opus"),
    ("Work out how to detect that the sidecar's routing quality has degraded "
     "in production, and design the alerting for it.", "opus"),
]

_NOOP_AGENT = "capture-noop"


async def _capture_one(user_request: str, model: str) -> List[str]:
    from claude_agent_sdk import (AgentDefinition, ClaudeAgentOptions, HookMatcher,
                                  ResultMessage, query)
    from claude_agent_sdk import ClaudeSDKError
    from claude_router.agents import DELEGATION_TOOLS

    seen: List[str] = []

    async def hook(inp: Dict[str, Any], tool_use_id: Any, ctx: Any) -> Dict[str, Any]:
        if inp.get("tool_name") not in DELEGATION_TOOLS:
            return {}
        tool_input = dict(inp.get("tool_input") or {})
        seen.append(tool_input.get("prompt") or tool_input.get("description") or "")
        # Redirect rather than deny: denying makes the orchestrator retry (measured
        # 8 turns, $0.18), and `continue_: False` is not honoured here either.
        tool_input["subagent_type"] = _NOOP_AGENT
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": tool_input}}

    options = ClaudeAgentOptions(
        model=model,
        # The orchestrator reads agent *descriptions* and echoes them into the
        # brief it writes -- a first pass whose description said "Records the
        # task. Does not perform it." leaked that wording into 4/15 captures and
        # inflated their difficulty. The description must therefore look like an
        # ordinary worker; the no-op lives in `prompt`, which the orchestrator
        # never sees.
        # `tools=[]` is load-bearing, not belt-and-braces. Restricting the
        # *orchestrator* to the delegation tool does nothing for the subagent,
        # which otherwise inherits the full tool set -- and with
        # bypassPermissions there is no prompt to stop it. A first pass relied on
        # the prompt alone ("Reply with exactly: RECORDED") and subagents ignored
        # it, editing claude_router/router.py and adding a test file to the repo
        # during a capture run. An instruction is not a sandbox.
        agents={_NOOP_AGENT: AgentDefinition(
            description="Delegated work. Selected by the router, not by you.",
            prompt="Reply with exactly: RECORDED. Do nothing else.",
            model="haiku", maxTurns=1, tools=[],
            disallowedTools=["Write", "Edit", "NotebookEdit", "Bash", "Agent", "Task"])},
        system_prompt={"type": "preset", "preset": "claude_code",
                       "append": "Delegate this task to a single subagent with "
                                 "subagent_type=%r, then stop. The subagent's reply "
                                 "is final -- do not verify or redo its work." % _NOOP_AGENT},
        allowed_tools=list(DELEGATION_TOOLS),
        permission_mode="bypassPermissions",
        max_turns=4,
    )
    options.hooks = {"PreToolUse": [HookMatcher(matcher="|".join(DELEGATION_TOOLS), hooks=[hook])]}

    cost = 0.0
    try:
        async for message in query(prompt=user_request, options=options):
            if isinstance(message, ResultMessage):
                cost = message.total_cost_usd or cost
    except ClaudeSDKError as e:
        # Hitting the turn cap is expected: the orchestrator often keeps going
        # after the no-op subagent replies, and the prompt was already captured
        # by then. Transport failures are not expected but are worth surviving
        # -- a 15-run capture should not be lost to one flaky subprocess.
        print("    (%s: %s)" % (type(e).__name__, str(e)[:80]), file=sys.stderr)
    print("    captured %d, $%.4f" % (len(seen), cost), file=sys.stderr)
    return seen


async def capture(model: str, refresh: bool = False) -> int:
    # Resumable and non-destructive: a capture run costs real money and can die
    # part-way (a session limit ended one at case 12), so already-captured cases
    # are kept and only the gaps are filled. `--refresh` forces a full recapture.
    have = {}
    if os.path.exists(CORPUS) and not refresh:
        have = {c["request"]: c for c in json.load(open(CORPUS))}
    corpus = []
    for i, (request, want) in enumerate(CASES, 1):
        if request in have:
            corpus.append(have[request])
            continue
        print("[%d/%d] %s" % (i, len(CASES), request[:60]), file=sys.stderr)
        prompts = await _capture_one(request, model)
        if not prompts:                          # one retry, then give up on this case
            print("    retrying", file=sys.stderr)
            prompts = await _capture_one(request, model)
        if not prompts:
            print("    NO DELEGATION -- skipped", file=sys.stderr)
            continue
        # One user request, one delegated task: take the first and keep the
        # pairing unambiguous rather than guessing which capture is which.
        corpus.append({"request": request, "want": want, "delegated": prompts[0],
                       "orchestrator": model})
    leaked = [c["request"] for c in corpus
              if re.search(r"record (this|the) task|do(es)? not perform|RECORDED", c["delegated"], re.I)]
    if leaked:
        print("\nWARNING: %d capture(s) echo the harness's own wording and are not "
              "representative:\n  %s" % (len(leaked), "\n  ".join(s[:60] for s in leaked)),
              file=sys.stderr)
    with open(CORPUS, "w") as f:
        json.dump(corpus, f, indent=2)
    print("\nwrote %d/%d cases to %s" % (len(corpus), len(CASES), CORPUS), file=sys.stderr)
    return 0 if len(corpus) == len(CASES) else 1


def score(verbose: bool = True, as_json: bool = False) -> int:
    from claude_router import RouterClient

    if not os.path.exists(CORPUS):
        print("no corpus; run --capture first", file=sys.stderr)
        return 2
    corpus = json.load(open(CORPUS))
    # Generous timeout on purpose: the client's 2.0 s default is tuned for short
    # user prompts and a delegated brief occasionally breaches it (measured: one
    # route over 2 s against a 115-776 ms norm). Here that would silently score a
    # fallback-to-Opus as a routing decision.
    client = RouterClient(on_error="raise", timeout=30.0)
    rows, hits = [], 0
    for case in corpus:
        delegated = client.route(case["delegated"])
        direct = client.route(case["request"])
        hit = delegated.tier == case["want"]
        hits += hit
        rows.append((case, delegated, direct, hit))

    lens = [len(c["delegated"]) for c, _, _, _ in rows]
    reqs = [len(c["request"]) for c, _, _, _ in rows]
    moved = sum(1 for _, d, t, _ in rows if d.tier != t.tier)

    if as_json:
        print(json.dumps({
            "hits": hits,
            "total": len(rows),
            "avg_request_chars": sum(reqs) // len(reqs),
            "avg_delegated_chars": sum(lens) // len(lens),
            "tier_changes": moved,
            "cases": [{
                "request": case["request"],
                "want": case["want"],
                "direct_tier": direct.tier,
                "direct_difficulty": direct.difficulty,
                "delegated_tier": delegated.tier,
                "delegated_difficulty": delegated.difficulty,
                "hit": hit,
            } for case, delegated, direct, hit in rows],
        }))
        return 0 if hits == len(rows) else 1

    if verbose:
        print("%-46s %-7s %-16s %-16s" % ("user request", "want", "as typed", "as delegated"))
        print("-" * 92)
        for case, delegated, direct, hit in rows:
            print("%-46s %-7s %-7s %-8.2f %-7s %-8.2f%s" % (
                case["request"][:44], case["want"], direct.tier, direct.difficulty,
                delegated.tier, delegated.difficulty, "" if hit else "  <-- MISS"))
        print("\n%d/%d correct on delegated phrasing" % (hits, len(rows)))
        print("prompt length: %d chars as typed -> %d as delegated (%.1fx)"
              % (sum(reqs) // len(reqs), sum(lens) // len(lens),
                 (sum(lens) / len(lens)) / (sum(reqs) / len(reqs))))
        print("%d/%d cases change tier between the two phrasings" % (moved, len(rows)))
    return 0 if hits == len(rows) else 1


RANK = {"haiku": 0, "sonnet": 1, "opus": 2}

# ---------------------------------------------------------------------------
# The candidate replacement for the difficulty score, kept here (not in the
# package) until the corpus is big enough to justify promoting it.
#
# Two independent findings drive it, both measured on the padding ladder in
# `--ladder`:
#   1. `difficulty` tracks prompt length. Holding the task fixed and pasting in
#      a 250-word essay walks it 1.18 -> 2.05, cheapest tier to most expensive.
#      Truncating to the first two sentences makes it flat (1.18/1.17/1.17/1.17).
#   2. A `choice` question separates tiers far better than the score does
#      (11/15 vs 3-4/15 on the first corpus), but is itself fooled by padding.
# Stripping fixes (1), the choice question fixes (2), and they compose.
# ---------------------------------------------------------------------------
FENCE = re.compile(r"```.*?```", re.S)
KIND_TIER = {"lookup": "haiku", "edit": "sonnet", "design": "opus"}


def strip_payload(prompt: str, sentences: int = 2) -> str:
    """Keep the ask, drop the pasted material it refers to.

    A stack trace, a file, a log or a quoted document is what the request is
    *about*, not what it asks for, and including it is what makes an easy task
    read as hard. Two sentences beat one (8/15 vs 4/15 on the first corpus).
    """
    text = FENCE.sub(" ", prompt).strip()
    return " ".join(re.split(r"(?<=[.!?])\s+", text)[:sentences]).strip()


# Candidate wordings. The four failures on the first corpus were all sonnet
# tasks ("explain", "debug", "summarize") called "design", so the variants
# differ in where they put explanation and diagnosis.
KIND_VARIANTS = {
    "v1-baseline": {
        "lookup": "find, read, list or report something that already exists; no judgement needed",
        "edit": "write or change code in a contained way, or explain something specific",
        "design": "design, architect, prove, evaluate trade-offs, or make a judgement call",
    },
    "v2-explain-is-edit": {
        "lookup": "find, read, list or report a fact that already exists; no reasoning needed",
        "edit": "make a contained change, write a test, explain how existing code works, "
                "or diagnose a specific bug -- the kind of task with one right answer",
        "design": "propose a new design, weigh trade-offs, prove something, or make a "
                  "recommendation that could reasonably go more than one way",
    },
    "v3-by-output": {
        "lookup": "the answer is a value copied out of the codebase",
        "edit": "the answer is a specific change, test, or explanation of existing behaviour",
        "design": "the answer is an argued recommendation among competing options",
    },
}


def _kind_questions(criteria):
    return {"kind": {"type": "choice",
                     "instructions": "What kind of work does `request` ask for?",
                     "criteria": criteria}}


def _split(corpus):
    """Stratified train/test split, deterministic, no shuffling library.

    Every third case of each tier goes to test. Tuning wording against the same
    cases you report on is how you get a number that does not survive contact
    with new data, so the headline is always the held-out score.
    """
    train, test = [], []
    seen = {}
    for case in sorted(corpus, key=lambda c: c["request"]):
        n = seen.get(case["want"], 0)
        seen[case["want"]] = n + 1
        (test if n % 3 == 2 else train).append(case)
    return train, test


def categories(sentences: int = 2) -> int:
    """Score each candidate wording on train and on held-out test."""
    from laya import load

    corpus = json.load(open(CORPUS))
    train, test = _split(corpus)
    print("corpus %d cases -> train %d / test %d" % (len(corpus), len(train), len(test)))
    for name, part in (("train", train), ("test", test)):
        counts = {}
        for c in part:
            counts[c["want"]] = counts.get(c["want"], 0) + 1
        print("  %-5s %s" % (name, counts))

    agent = load()

    def ask(questions, text):
        return agent.system_one({"request": text}, questions)["answers"]["kind"]["choice"]

    print("\n%-22s %-14s %-14s" % ("wording", "train", "TEST (held out)"))
    print("-" * 54)
    results = {}
    for name, criteria in KIND_VARIANTS.items():
        questions = _kind_questions(criteria)
        scores = []
        for part in (train, test):
            hits = sum(KIND_TIER[ask(questions, strip_payload(c["delegated"], sentences))] == c["want"]
                       for c in part)
            scores.append((hits, len(part)))
        results[name] = scores
        print("%-22s %-14s %-14s" % (name, "%d/%d" % scores[0], "%d/%d" % scores[1]))

    best = max(results, key=lambda k: results[k][1][0] / max(results[k][1][1], 1))
    print("\nbest on held-out test: %s (%d/%d)" % (best, *results[best][1]))

    print("\nheld-out misses for %s:" % best)
    questions = _kind_questions(KIND_VARIANTS[best])
    for c in test:
        got = ask(questions, strip_payload(c["delegated"], sentences))
        if KIND_TIER[got] != c["want"]:
            print("  want %-7s got %-7s (%s)  %s" % (c["want"], KIND_TIER[got], got, c["request"][:44]))
    return 0


def ladder() -> int:
    """Hold the task fixed, vary how much pasted material comes with it."""
    from laya import load

    essay = (" ".join(c["delegated"] for c in json.load(open(CORPUS))[:3]))[:1400]
    base = "Find the word 'reconciliation' in this text and tell me how many times it appears."
    steps = [("bare", base)]
    for label, words in (("+20 words", 20), ("+120 words", 120), ("+all", 10000)):
        steps.append((label, base + "\n\n" + " ".join(essay.split()[:words])))

    agent = load()
    questions = _kind_questions(KIND_VARIANTS["v1-baseline"])

    def sc(t):
        from claude_router.router import ROUTER_QUESTIONS
        return float(agent.system_one({"request": t}, {"difficulty": ROUTER_QUESTIONS["difficulty"]}
                                      )["answers"]["difficulty"]["score"])

    def kd(t):
        return agent.system_one({"request": t}, questions)["answers"]["kind"]["choice"]

    print("The task is EASY at every step; only the pasted text grows.\n")
    print("%-24s %s" % ("", "  ".join("%-11s" % l for l, _ in steps)))
    print("%-24s %s" % ("difficulty, raw", "  ".join("%-11.2f" % sc(p) for _, p in steps)))
    print("%-24s %s" % ("difficulty, stripped", "  ".join("%-11.2f" % sc(strip_payload(p)) for _, p in steps)))
    print("%-24s %s" % ("kind, raw", "  ".join("%-11s" % kd(p) for _, p in steps)))
    print("%-24s %s" % ("kind, stripped", "  ".join("%-11s" % kd(strip_payload(p)) for _, p in steps)))
    return 0




def _spearman(a: List[float], b: List[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos
        return out
    ra, rb, n = ranks(a), ranks(b), len(a)
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def tune() -> int:
    """Ask whether *any* pair of cuts could work, and what the score tracks."""
    from claude_router import RouterClient

    corpus = json.load(open(CORPUS))
    client = RouterClient(on_error="raise", timeout=30.0)
    rows = []
    for case in corpus:
        rows.append((len(case["delegated"]), client.route(case["delegated"]).difficulty,
                     RANK[case["want"]], client.route(case["request"]).difficulty,
                     len(case["request"])))

    lengths = [r[0] for r in rows]
    delegated = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    typed = [r[3] for r in rows]
    typed_len = [r[4] for r in rows]

    print("What the difficulty score tracks (Spearman rho)")
    print("  as typed:     length %+.2f   labelled tier %+.2f" %
          (_spearman(typed_len, typed), _spearman(labels, typed)))
    print("  as delegated: length %+.2f   labelled tier %+.2f" %
          (_spearman(lengths, delegated), _spearman(labels, delegated)))
    print("\nPer-tier spread, as delegated")
    for tier, rank in sorted(RANK.items(), key=lambda kv: kv[1]):
        diffs = sorted(r[1] for r in rows if r[2] == rank)
        chars = sorted(r[0] for r in rows if r[2] == rank)
        print("  %-7s difficulty %.2f - %.2f   brief %d - %d chars"
              % (tier, diffs[0], diffs[-1], chars[0], chars[-1]))

    grid = [x / 100 for x in range(80, 260)]
    best = (0, 0.0, 0.0)
    for cheap in grid:
        for opus in grid:
            if opus < cheap:
                continue
            ok = sum(("opus" if d >= opus else "sonnet" if d >= cheap else "haiku")
                     == ("haiku", "sonnet", "opus")[w] for _, d, w, _, _ in rows)
            if ok > best[0]:
                best = (ok, cheap, opus)
    print("\nbest achievable with ANY pair of cuts: %d/%d (cheap=%.2f opus=%.2f)"
          % (best[0], len(rows), best[1], best[2]))
    print("-- fitted on %d points with 2 free parameters, so treat it as a ceiling,"
          "\n   not an operating point." % len(rows))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval_delegated")
    ap.add_argument("--capture", action="store_true", help="regenerate the corpus (costs money)")
    ap.add_argument("--model", default="sonnet", help="orchestrator model for --capture")
    ap.add_argument("--refresh", action="store_true", help="recapture cases already on disk")
    ap.add_argument("--tune", action="store_true", help="can any cuts work? what does the score track?")
    ap.add_argument("--categories", action="store_true",
                    help="score the strip+choice candidate, train vs held-out test")
    ap.add_argument("--ladder", action="store_true",
                    help="hold the task fixed, vary pasted length; shows the length effect")
    ap.add_argument("--json", action="store_true", help="print score results as JSON")
    args = ap.parse_args(argv)
    if args.capture:
        return asyncio.run(capture(args.model, args.refresh))
    if args.tune:
        return tune()
    if args.categories:
        return categories()
    if args.ladder:
        return ladder()
    return score(as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
