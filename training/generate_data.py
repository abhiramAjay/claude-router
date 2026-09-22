"""Generate labelled delegated briefs for training the routing head.

Works against any OpenAI-compatible chat endpoint, so the free tiers all work:

    export GEN_BASE_URL=https://api.groq.com/openai/v1
    export GEN_API_KEY=gsk_...
    export GEN_MODEL=llama-3.3-70b-versatile
    python training/generate_data.py --n 1000

    # presets save you the base_url lookup
    python training/generate_data.py --provider groq      --model llama-3.3-70b-versatile --n 400
    python training/generate_data.py --provider openrouter --model deepseek/deepseek-chat --n 400
    python training/generate_data.py --provider gemini    --model gemini-2.0-flash        --n 400

Standard library only -- no SDK for any provider, since the point is to mix
several free ones. Output is appended to `training/briefs.jsonl`, deduplicated
by brief text, so you can run it repeatedly with different providers and models
and the corpus only grows.

Why the data looks the way it does
----------------------------------
The failure this whole exercise is trying to fix is that difficulty scoring
tracks prompt *length*: holding the task fixed and pasting in a 250-word essay
walked laya 1.18 -> 2.05, cheapest tier to most expensive. Text-stripping
patches that but breaks on payloads with no sentence punctuation -- a stack
trace or a log dump survives the stripper intact.

So the fix belongs in the data, not in a regex. Every request here samples a
(tier, payload size) pair independently, which forces the set to contain plenty
of trivial tasks carrying enormous payloads and plenty of hard tasks stated in
one line. A model trained on that cannot use length as a shortcut, because
length carries no information about the label.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "briefs.jsonl")

PROVIDERS = {
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai",
    "cerebras":   "https://api.cerebras.ai/v1",
    "together":   "https://api.together.xyz/v1",
}

# The three routing tiers, described by what the *work* is -- never by how much
# text describes it.
TIERS = {
    "lookup": "find, read, list or report something that already exists in the "
              "codebase. One tool call and a verbatim answer. No judgement.",
    "edit":   "a contained change, a test, a bug diagnosis, or an explanation of "
              "how existing code works. Bounded, with one right answer.",
    "design": "design, architecture, trade-off analysis, a proof, or a "
              "recommendation that could reasonably go more than one way.",
}

# Sampled independently of the tier. This is the whole point: a `lookup` is just
# as likely to carry a 300-line stack trace as a `design` is to be one sentence.
PAYLOADS = [
    ("none",        "no pasted material at all; just the instruction"),
    ("short",       "a few lines of pasted code or config"),
    ("stacktrace",  "a full stack trace or error dump, 20-40 lines, NO sentence "
                    "punctuation -- raw frames and paths"),
    ("logs",        "60+ lines of log output, timestamps and levels, no prose"),
    ("bigfile",     "a large pasted source file, 100+ lines"),
    ("transcript",  "several paragraphs of quoted prose: a spec, an issue "
                    "comment thread, or documentation"),
]

STACKS = ["a Python FastAPI service", "a TypeScript React app", "a Go CLI tool",
          "a Rust parser", "a Django monolith", "a Node/Express API",
          "a Kotlin Android app", "a C++ game engine", "a Terraform module",
          "a Ruby on Rails app", "a Java Spring service", "a Next.js frontend",
          "a data pipeline in PySpark", "an Elixir Phoenix app", "a SwiftUI iOS app"]

PROMPT = """You are generating training data for a model that decides which AI \
model should handle a delegated coding task.

Write {n} DIFFERENT examples of the prompt an AI coding orchestrator hands to a \
subagent. These are internal briefs, not user messages: they name files, give \
repo context, and state acceptance criteria, exactly as a coding agent would \
write when delegating.

Every example in this batch must be for {stack} and must be labelled "{tier}":
{tier_desc}

Every example in this batch must include: {payload_desc}.

CRITICAL: the amount of pasted material must NOT reflect how hard the task is. \
In this batch the task is "{tier}" and the payload is "{payload}" -- keep both \
exactly as specified even when that feels mismatched. A trivial lookup wrapped \
in a 200-line log dump is a deliberately wanted example.

Return ONLY a JSON array, no markdown fence, no commentary:
[{{"brief": "<the full delegated prompt, payload included, \\n for newlines>", \
"tier": "{tier}"}}]
"""


def post(base_url: str, api_key: str, model: str, prompt: str, timeout: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": 8000,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse(text: str) -> List[Dict[str, Any]]:
    """Pull the JSON array out, tolerating fences and stray prose around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        start, end = text.index("["), text.rindex("]") + 1
    except ValueError:
        return []
    try:
        rows = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    out = []
    for r in rows:
        if isinstance(r, dict) and r.get("brief") and r.get("tier") in TIERS:
            out.append({"brief": str(r["brief"]), "tier": r["tier"]})
    return out


def load_seen() -> set:
    if not os.path.exists(OUT):
        return set()
    seen = set()
    with open(OUT) as f:
        for line in f:
            try:
                seen.add(hashlib.sha1(json.loads(line)["brief"].encode()).hexdigest())
            except Exception:
                pass
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="generate_data")
    ap.add_argument("--provider", choices=sorted(PROVIDERS), help="base_url preset")
    ap.add_argument("--base-url", default=os.environ.get("GEN_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("GEN_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("GEN_MODEL"))
    ap.add_argument("--n", type=int, default=200, help="target number of new briefs")
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between calls (rate limits)")
    args = ap.parse_args(argv)

    base_url = PROVIDERS[args.provider] if args.provider else args.base_url
    if not (base_url and args.api_key and args.model):
        ap.error("need --provider/--base-url, --api-key and --model "
                 "(or GEN_BASE_URL / GEN_API_KEY / GEN_MODEL)")

    seen = load_seen()
    print("%d brief(s) already on disk" % len(seen), file=sys.stderr)
    rng = random.Random()
    written = fails = 0
    t0 = time.time()

    with open(OUT, "a") as out:
        while written < args.n:
            tier = rng.choice(list(TIERS))
            payload, payload_desc = rng.choice(PAYLOADS)
            prompt = PROMPT.format(n=args.per_call, stack=rng.choice(STACKS),
                                   tier=tier, tier_desc=TIERS[tier],
                                   payload=payload, payload_desc=payload_desc)
            try:
                rows = parse(post(base_url, args.api_key, args.model, prompt, args.timeout))
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                fails += 1
                print("  call failed (%s: %s)" % (type(e).__name__, str(e)[:70]), file=sys.stderr)
                if fails > 12 and not written:
                    print("giving up: nothing is coming back", file=sys.stderr)
                    return 1
                time.sleep(min(2 ** min(fails, 5), 30))
                continue

            new = 0
            for r in rows:
                h = hashlib.sha1(r["brief"].encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                r["payload"] = payload
                r["source"] = args.model
                out.write(json.dumps(r) + "\n")
                new += 1
            out.flush()
            written += new
            print("  +%d (%d/%d) %s/%s" % (new, written, args.n, tier, payload), file=sys.stderr)
            time.sleep(args.sleep)

    print("\nwrote %d new briefs in %.0fs -> %s (%d total)"
          % (written, time.time() - t0, OUT, len(seen)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
