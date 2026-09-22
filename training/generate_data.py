"""Generate labelled delegated briefs for training the routing head.

Works against any OpenAI-compatible chat endpoint, so the free tiers all work:

    export GEN_BASE_URL=https://api.groq.com/openai/v1
    export GEN_API_KEY=gsk_...
    export GEN_MODEL=<id from --list-models>
    python training/generate_data.py --n 1000

    # presets save you the base_url lookup; ask the provider which models exist
    # rather than trusting a hardcoded name, because they churn
    python training/generate_data.py --provider groq --api-key ... --list-models
    python training/generate_data.py --provider groq --api-key ... --model <id> --n 400

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

# Some providers sit behind a WAF that 403s urllib's default User-Agent
# ("Python-urllib/3.x") while the identical request from curl succeeds.
# Measured against Groq: 403 on /models until this header is set.
UA = "claude-router-traindata/0.1"

PROVIDERS = {
    "openai":     "https://api.openai.com/v1",
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

SEEDED_PROMPT = """You are generating training data for a model that decides which \
AI model should handle a delegated coding task.

Below are REAL examples of briefs that a coding orchestrator actually wrote when \
delegating to a subagent. Study their voice, structure, length and messiness:

{examples}

Now write {n} DIFFERENT examples in that same style -- not the style of a tidy \
template. Real briefs vary: some open with repo context, some dive straight in, \
some ramble, some are terse. Match that variety rather than a fixed \
Repo/File/Context/Task/Acceptance layout.

Every example must be for {stack} and must be labelled "{tier}":
{tier_desc}

Every example must include: {payload_desc}.

CRITICAL: the amount of pasted material must NOT reflect how hard the task is. \
In this batch the task is "{tier}" and the payload is "{payload}" -- keep both \
exactly as specified even when that feels mismatched. A trivial lookup wrapped \
in a 200-line log dump is a deliberately wanted example.

Return ONLY a JSON array, no markdown fence, no commentary:
[{{"brief": "<the full delegated prompt, payload included, \\n for newlines>", \
"tier": "{tier}"}}]
"""

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


def post(base_url: str, api_key: str, model: str, prompt: str, timeout: float,
         max_tokens: int = 4000) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA,
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


def list_models(base_url: str, api_key: str, timeout: float) -> int:
    """Print the model ids this key can actually use.

    Model names churn fast -- a name that was current when this file was
    written may already be retired -- so ask the provider rather than trusting
    any list, including the examples in the docstring above.
    """
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + api_key, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("%s %s -- is the key right, and does this provider expose /models?"
              % (e.code, e.reason), file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as e:
        print("could not reach %s (%s)" % (base_url, type(e).__name__), file=sys.stderr)
        return 1
    ids = sorted(str(m.get("id", "?")).split("models/")[-1]
                 for m in (data.get("data") or data.get("models") or []))
    if not ids:
        print("no models listed; raw response: %s" % json.dumps(data)[:300], file=sys.stderr)
        return 1
    for i in ids:
        print(i)
    print("\n%d model(s). Pick one and pass it as --model." % len(ids), file=sys.stderr)
    return 0


def load_seeds() -> Dict[str, List[str]]:
    """Real briefs to few-shot the generator with, keyed by tier.

    Only the TRAINING split of the real corpus is ever used. The held-out cases
    must never reach the generator: briefs written in imitation of a test case
    would make the final score measure memorisation instead of transfer, and the
    whole point of keeping real data as the test set is that it is independent.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_delegated", os.path.join(os.path.dirname(HERE), "eval_delegated.py"))
    ed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ed)

    corpus = json.load(open(os.path.join(os.path.dirname(HERE), "delegated_corpus.json")))
    train, test = ed._split(corpus)
    held_out = {c["request"] for c in test}
    assert not (held_out & {c["request"] for c in train}), "split leaked"

    tier_kind = {"haiku": "lookup", "sonnet": "edit", "opus": "design"}
    seeds: Dict[str, List[str]] = {"lookup": [], "edit": [], "design": []}
    for c in train:
        seeds[tier_kind[c["want"]]].append(c["delegated"])
    print("seeding from %d real briefs (%s); %d held-out cases withheld"
          % (len(train), {k: len(v) for k, v in seeds.items()}, len(test)), file=sys.stderr)
    return seeds


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
    ap.add_argument("--per-call", type=int, default=5)
    ap.add_argument("--seed-from-real", action="store_true",
                    help="few-shot the generator with REAL briefs from the training "
                         "split, so it imitates the real distribution instead of "
                         "inventing a tidy one. Held-out cases are never used.")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="lower this if a provider rate-limits on tokens/minute")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between calls (rate limits)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model ids this key can use, then exit")
    ap.add_argument("--out", default=None,
                    help="output file (default training/briefs.jsonl). Give each "
                         "provider its own when running them concurrently: a 2 KB "
                         "append is not atomic, so sharing one file can interleave "
                         "lines. train_head.py globs training/briefs*.jsonl.")
    args = ap.parse_args(argv)

    base_url = PROVIDERS[args.provider] if args.provider else args.base_url
    if args.list_models:
        if not (base_url and args.api_key):
            ap.error("--list-models needs --provider/--base-url and --api-key")
        return list_models(base_url, args.api_key, args.timeout)
    if not (base_url and args.api_key and args.model):
        ap.error("need --provider/--base-url, --api-key and --model "
                 "(or GEN_BASE_URL / GEN_API_KEY / GEN_MODEL)")

    global OUT
    if args.out:
        OUT = args.out
    seeds = load_seeds() if args.seed_from_real else {}
    seen = load_seen()
    print("%d brief(s) already on disk" % len(seen), file=sys.stderr)
    rng = random.Random()
    written = fails = 0
    t0 = time.time()

    with open(OUT, "a") as out:
        while written < args.n:
            tier = rng.choice(list(TIERS))
            payload, payload_desc = rng.choice(PAYLOADS)
            if seeds:
                # Show same-tier exemplars: the generator should copy how a real
                # brief of THIS kind reads, not just the house style in general.
                pool = seeds.get(tier) or [b for v in seeds.values() for b in v]
                chosen = rng.sample(pool, min(2, len(pool)))
                examples = "\n\n".join(
                    "--- REAL EXAMPLE %d ---\n%s" % (i + 1, b[:1500])
                    for i, b in enumerate(chosen))
                prompt = SEEDED_PROMPT.format(
                    examples=examples, n=args.per_call, stack=rng.choice(STACKS),
                    tier=tier, tier_desc=TIERS[tier],
                    payload=payload, payload_desc=payload_desc)
            else:
                prompt = PROMPT.format(n=args.per_call, stack=rng.choice(STACKS),
                                       tier=tier, tier_desc=TIERS[tier],
                                       payload=payload, payload_desc=payload_desc)
            try:
                rows = parse(post(base_url, args.api_key, args.model, prompt,
                                  args.timeout, args.max_tokens))
            except urllib.error.HTTPError as e:
                fails += 1
                # Free tiers rate-limit aggressively and most of them say for how
                # long. Honouring Retry-After beats exponential guessing: it both
                # backs off enough and stops sleeping longer than necessary.
                wait = 0.0
                try:
                    wait = float(e.headers.get("retry-after") or 0)
                except (TypeError, ValueError):
                    wait = 0.0
                if not wait:
                    wait = min(2 ** min(fails, 5), 30)
                # 413 is Groq's way of saying the request would exceed the
                # remaining tokens-per-minute allowance -- the same request
                # succeeds once the window rolls over, so it is a rate limit
                # wearing a different status code, not a malformed request.
                # 520-527 are Cloudflare's own edge errors (524 = origin timed
                # out). They say nothing about the request, only that a slow
                # model outran the proxy, so retrying with fewer briefs per call
                # is the right response.
                if e.code not in (408, 409, 413, 429, 500, 502, 503, 529) and not (520 <= e.code <= 527):
                    # The body is where providers put the actionable part -- the
                    # replacement model id for a retired one, or "no credits".
                    # Printing only the status hides it.
                    try:
                        detail = e.read().decode("utf-8", "replace")[:400]
                    except Exception:
                        detail = ""
                    print("  %s %s -- not retryable, stopping\n  %s"
                          % (e.code, e.reason, detail), file=sys.stderr)
                    return 1
                print("  %s, waiting %.0fs" % (e.code, wait), file=sys.stderr)
                if fails > 40 and not written:
                    print("giving up: nothing is coming back", file=sys.stderr)
                    return 1
                time.sleep(wait)
                continue
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
