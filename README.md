# claude-router

Picks a Claude model per prompt by complexity, using [laya](https://pypi.org/project/laya)
as a local classifier. No API call and no tokens are spent to make the decision:
laya is a 421M ModernBERT that answers typed questions in one forward pass,
~75 ms on CPU (M-series, after a one-time ~90 s model download).

## Setup

The classifier is heavy (951 MB venv, ~20 s to load) but the *decision* is tiny.
So it runs once as a background sidecar, and your projects install a
standard-library-only client — **13 MB, zero dependencies, no torch**.

### 1. Host the sidecar (once per machine)

```sh
python3 -m venv ~/.venvs/laya
~/.venvs/laya/bin/pip install ~/ideaProjects/claude-router[server]
sh ~/ideaProjects/claude-router/launchd/install.sh      # start at login, restart if it dies
curl -s http://127.0.0.1:8129/health       # {"ok": true, "loaded": true, ...}
```

Loopback only, no auth — don't expose the port. Logs land in
`~/Library/Logs/laya-routerd.log`. To run it by hand instead:
`~/.venvs/laya/bin/laya-routerd --verbose`.

The plist sets `ProcessType Background` + `Nice 5` so the daemon never competes
with your real work. That costs startup — 98 s under launchd vs 20 s foreground —
but it happens at login, long before you need it, and steady-state routing is
barely affected (85 ms vs 70 ms). Drop both keys if you want the fast boot.

```sh
launchctl print gui/$(id -u)/com.claude-router.laya-routerd   # state, pid, exit code
launchctl bootout gui/$(id -u)/com.claude-router.laya-routerd # stop and uninstall
```

### 2. In each project

```sh
pip install ~/ideaProjects/claude-router        # no extras: stdlib only
```

```python
from claude_router import RouterClient

router = RouterClient()
route = router.route(prompt)

client.messages.create(**route.messages_kwargs(), max_tokens=16000,
                       messages=[{"role": "user", "content": prompt}])
```

`messages_kwargs()` emits per-model-correct params: Haiku 4.5 gets neither
`thinking` nor `output_config.effort` (it rejects both); Sonnet 5 and Opus 5 get
adaptive thinking plus an effort level.

### If the sidecar is down

`route()` returns `claude-opus-5` with `reason="router unavailable (...)"`. You
stop saving money; nothing breaks. Override per client or by environment:

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_ROUTER_URL` | `http://127.0.0.1:8129` | where the sidecar lives |
| `CLAUDE_ROUTER_FALLBACK` | `claude-opus-5` | model used when it is unreachable |

`RouterClient(on_error="raise")` surfaces the failure instead, if you would
rather find out immediately than quietly pay Opus rates.

### Measured on this machine

| | |
|---|---|
| sidecar startup, foreground | 19.8 s, once |
| sidecar startup, under launchd | 98.5 s, once at login (throttled by `ProcessType Background`) |
| route, over HTTP from a project | 70 ms (85 ms under launchd) |
| route, in-process | 75 ms |
| project venv | 13 MB, 0 dependencies |
| sidecar venv | 951 MB (590 MB torch) + 807 MB shared model cache |

## Tiers

| Tier | Model | $/1M in | $/1M out | Chosen when |
|---|---|---|---|---|
| cheap | `claude-haiku-4-5` | 1.00 | 5.00 | difficulty below the domain's cut |
| mid | `claude-sonnet-5` | 2.00 | 10.00 | difficulty 1.20–1.95 |
| high | `claude-opus-5` | 5.00 | 25.00 | difficulty ≥ 1.95, or the high-stakes guard fires |

Effort is the second lever: Sonnet `medium`/`high` at 1.60, Opus `high`/`xhigh` at 2.15.

## What laya is and isn't used for

Measured over the labeled cases in `tests/test_routing.py`:

- **`difficulty` works.** Monotone and well separated: 0.08 (`"hi there!"`) →
  1.09 (rename a variable) → 1.83 (add pagination + tests) → 2.26 (design a
  distributed rate limiter). This drives the tier choice.
- **`domain` mostly works** — chitchat, factual_lookup, math_or_logic and
  data_analysis land correctly; prose about databases gets called `code`. Used
  only to raise the cheap-tier cut for trivia and arithmetic, which laya scores
  around 1.35 because the *question* reads substantive even when the *answer* is
  a lookup.
- **Stakes does NOT work, and is not asked of laya.** Both the stock
  `router_questions()` preset (`is_sensitive`) and a hand-tuned `stakes` score
  returned ~1.0 for everything — 0.96 for `"hi there!"` and 0.97 for
  `"drop the users table in prod"`. The score head emits a prior, not a judgment.
  High-stakes escalation is a deterministic regex (`HIGH_STAKES`) instead, and it
  floors the route at Opus regardless of difficulty.
- laya also warns that this checkpoint's choice-head temperatures are outside its
  calibrated range, so treat `confidence` on choice questions as unreliable.

## Known limits

- At difficulty ≈1.45 in `domain=code`, laya cannot separate `"what does the -p
  flag do in mkdir"` from `"summarize gRPC vs REST for our API docs"`. The cut
  sits on the side that overspends rather than underspends; the eval accepts
  either tier for that case.
- The guard is a keyword list, so it is precise but not complete — it catches
  `"deploy"` only when `prod` precedes it, and knows nothing about your domain.
  Extend `HIGH_STAKES`, or pass your own via `ClaudeRouter(escalate_pattern=...)`.
- `example.py` has not been run against the live API, so `messages_kwargs()` is
  still unverified — the routing, the eval and the emitted request params are all
  checked, but nothing has put them on the wire. (The subagent spike below *does*
  reach the API, via the Claude Code CLI's own credentials rather than
  `ANTHROPIC_API_KEY`; it exercises the Agent SDK's request path, not this one.)

## Subagent routing (Claude Agent SDK)

`claude_router/agents.py` puts the router inside the Claude Agent SDK -- which is
Claude Code as a library, so there is nothing to clone and the decision layer is
the only code worth owning.

```sh
pip install ~/ideaProjects/claude-router[agents]
python spike_subagents.py --dry-run     # routing table only; no API call, no tokens
python spike_subagents.py               # full run; needs credentials
```

```python
from claude_agent_sdk import query
from claude_router.agents import DELEGATION_INSTRUCTION, TaskRouter, routing_options

router = TaskRouter()
options = routing_options(router=router, system_prompt={
    "type": "preset", "preset": "claude_code", "append": DELEGATION_INSTRUCTION})
async for message in query(prompt=..., options=options):
    ...
print(router.summary())      # what laya chose, and why
```

### In the interactive CLI (no API key)

The same routing works in the `claude` you type into, via a command hook, so it
runs on your Claude Code login rather than `ANTHROPIC_API_KEY`. This repo's
`.claude/settings.json` enables it for this project:

```json
{"hooks": {"PreToolUse": [{"matcher": "Agent|Task", "hooks": [{"type": "command", "timeout": 10,
  "command": "PYTHONPATH=\"$CLAUDE_PROJECT_DIR\" python3 -m claude_router.cc_hook"}]}]}}
```

`claude_router/cc_hook.py` sets the Agent tool's `model` (haiku / sonnet / opus)
and leaves `subagent_type` alone, so Explore, Plan and project agents keep their
tools. It cannot set effort, which the CLI tool does not take per call, and it
cannot route the main conversation, only what that conversation delegates. If the
sidecar is down or slow (>5 s), the hook exits silently and Claude Code uses its
own default, not Opus. Decisions are logged to `~/Library/Logs/laya-cc-hook.jsonl`.

An idle sidecar's first forward pass takes longer than that 5 s, so the settings
also run `cc_hook --warm` on SessionStart: it detaches (~40 ms to return) and
sends one throwaway route so the first real delegation is warm. `/health` would
not work, because it never touches the model.

### Shared context pool

A subagent starts with only its brief, so a task that depends on what the user
said, or on what another agent found, is only as good as that brief.
`claude_router/pool.py` keeps a per-session pool at `~/.claude/laya-pool/`: user
messages (UserPromptSubmit), briefs and results (PostToolUse on `Agent`, plus
SubagentStop for background agents), deduplicated. `cc_hook --record` fills it,
and the routing hook appends it to every new brief *after* routing, so the router
still judges the task and not the pasted context.

The whole pool goes in, capped at ~12k chars: user messages always, then agent
results newest first, with a note when older ones are dropped. Within a session
that is small enough that including everything beats retrieving. laya retrieval
is the next step for when the cap starts to bite, and it is unmeasured.

Verified end to end: a second Explore agent whose brief omitted a value found by
the first answered correctly, citing the first agent's result, and read only the
one file it needed. `CLAUDE_ROUTER_POOL=0` turns it off; pools older than 7 days
are pruned on SessionStart.

Known gap: the main conversation is not told the pool exists, so it may doubt a
subagent that cites earlier results. The orchestrator agent's instructions
mention it; plain sessions do not.

### Why subagents and not `set_model()`

`ClaudeSDKClient.set_model()` exists, and it is the wrong lever. Prompt caches are
model-scoped and a coding thread resends its whole history every turn, so switching
mid-session forfeits the cache -- the same 2.4x penalty measured for `chat.py`
below, worse here because agent histories are bigger. A subagent starts on a fresh
context, so choosing its model costs nothing and decides the whole task.

### Where the decision is taken

`AgentDefinition.model`/`.effort` are static and `SubagentStart` can only append
context, so routing happens one step earlier: a `PreToolUse` hook on the delegation
tool reads the delegated prompt, asks laya, and sets `subagent_type` via
`updatedInput` before the subagent spawns. There is one agent per reachable
(tier, effort) pair -- five, not three -- so effort survives as a lever.

### Measured, Claude Code 2.1.x / SDK 0.2.157

Three delegated tasks in one run. `model_usage` is keyed by the model that actually
served the tokens, so this is the routing landing, not just the hook firing:

| | |
|---|---|
| `claude-haiku-4-5` | 1009 in / 643 out, $0.0256 |
| `claude-sonnet-5` | 10 in / 6377 out, $0.1751 |
| `claude-opus-5` | 8 in / 7542 out, $0.3457 |
| routing overhead | ~130 ms per delegated task, local, 0 tokens |

### Three things the spike found

- **The delegation tool is `Agent`, not `Task`.** `subagent_type` still appears in
  the CLI binary and the SDK docs, but a hook matching `Task` never fires on 2.1.x
  -- silently, with every subagent on the default model. `DELEGATION_TOOLS` matches
  both names; if a future build renames it again, the symptom is again silence.
- **The model usually omits `subagent_type`.** It calls the tool with only
  `description` and `prompt` and lets the type default, so the hook injects the key
  rather than rewriting it.
- **It will not delegate if it can do the work itself.** Given `Read`/`Glob`/`Grep`
  alongside the delegation tool, the run made zero delegations and spent $0.78
  entirely on the orchestrator, instruction to delegate notwithstanding. The spike
  allows the delegation tool only.

### Delegated phrasing breaks the calibration

The mechanism works. The *classifier* does not transfer to it, and that was worth
measuring before building anything on top.

`eval_delegated.py` captures what an orchestrator actually hands a subagent: it
runs a real Claude Code session over each user request, records the brief, and
redirects the subagent to a no-op Haiku agent so the paraphrase is paid for and
the work is not. 40 cases (12 lookup / 16 contained / 12 design), cached in
`delegated_corpus.json`, weighted toward the boundary where the classifier fails.

```sh
python eval_delegated.py              # score the saved corpus (free, no API)
python eval_delegated.py --ladder     # the length effect, isolated
python eval_delegated.py --categories # tune the candidate, train vs held-out test
python eval_delegated.py --capture    # regenerate; ~$0.15/case, resumable
```

**The difficulty score is substantially a length proxy.** Hold the task fixed and
paste in more text, and it walks the whole range:

| what laya sees | bare task | +20 words | +120 words | +full essay |
|---|---|---|---|---|
| raw prompt | 1.18 | 1.83 | 1.99 | 2.05 |
| first 2 sentences only | 1.18 | 1.17 | 1.17 | 1.17 |

The task never changed. That is fatal in normal use, where prompts carry stack
traces, logs and file contents. Counting a word in a pasted essay (2.19) outscores
proving that every finite integral domain is a field (2.00).

Two independent repairs, each fixing a different half:

- **Strip the payload.** Route the ask, not the material it refers to. Makes the
  ladder flat. Two sentences beat one.
- **Ask a `choice` question, not a `score`.** `lookup` / `edit` / `design` instead
  of "how hard is this, 0-3". Separates tiers far better, but is itself fooled by
  padding, so it needs the stripping.

Measured on the 40 cases, against the tiers as labelled:

| | held-out 13 | all 40 | overspends | picks the cheap tier |
|---|---|---|---|---|
| current: difficulty score, raw | 8/13 | **20/40** | 17 | **1 of 40** |
| strip + category | 10/13 | **27/40** | 10 | 10 of 40 |

The cheap-tier column is the real story: the shipped router reaches Haiku **once in
forty** delegated tasks, so the cascade collapses to Sonnet-and-Opus and saves
almost nothing. The replacement reaches it ten times.

**Tuning the category wording did not help.** Three wordings, scored on 27 training
cases and 13 held out:

```
wording                train          TEST (held out)
v1-baseline            17/27          10/13
v2-explain-is-edit     19/27          10/13
v3-by-output           20/27          10/13
```

Train improves, held-out does not move at all. The gain was fitting the training
cases, which is exactly what the split exists to catch. Keep v1.

### Where that leaves it

68% on 40 cases is better than 50% but is not good enough to ship, and the
remaining errors are systematic: the replacement over-picks `design`, choosing
Opus 21 times where 12 are warranted. The failures are consistently "explain",
"diagnose" and "summarize" tasks read as design work.

Worth weighing before spending more on wording: **laya's advantage was never cost,
it was latency.** It is free and ~100 ms. But a routing decision on a ~210-token
brief costs about **$0.00024** on Haiku 4.5 -- under 1% of the $0.03-$0.35 task it
is routing. If a real model classifies materially better, the ~1 s it adds is the
only thing being traded, and the hook in `claude_router/agents.py` is unchanged
either way. That comparison has not been run.

### Training a head instead of tuning wording

Wording tuning plateaued (three variants, identical held-out scores), so the
remaining option is to learn the boundary from data. laya ships a runtime only --
`predict` and `system_one`, no `fit` -- so the checkpoint cannot be fine-tuned.
It does expose `embed_fn_from_agent`, the mean-pooled encoder at **~38 ms per
prompt**, which is enough: freeze laya as a feature extractor and train a small
classifier on top. That keeps every property that made laya attractive (local,
~100 ms, $0 per decision, offline) and moves the tier boundary from prompt
wording into data.

```sh
pip install ~/ideaProjects/claude-router[server,head]

# 1. generate labelled briefs -- any OpenAI-compatible endpoint, so free tiers work
python training/generate_data.py --provider groq       --model llama-3.3-70b-versatile --api-key ... --n 400
python training/generate_data.py --provider openrouter --model deepseek/deepseek-chat   --api-key ... --n 400
python training/generate_data.py --provider gemini     --model gemini-2.0-flash         --api-key ... --n 400

# 2. train, and score against the 40 REAL briefs
python training/train_head.py
python training/train_head.py --curve     # does more data still help?
```

Mixing providers is deliberate: briefs from one model share its habits, and a
head trained on them learns the habits rather than the task.

**Why the generator samples payload size independently of the label.** Stripping
text was a patch for the length bias and it breaks on the payloads that matter --
a stack trace or a log dump has no sentence punctuation, so the stripper removes
nothing and the bias returns in full. Fixing it in the data instead means the set
contains trivial lookups wrapped in 200-line log dumps and one-line design
questions, so length carries no information about the label and the model cannot
use it as a shortcut. `claude_router/head.py` therefore embeds the brief raw.

**The evaluation is deliberately adversarial to the generator.** Training data is
synthetic; the test set is the 40 briefs captured from real Claude Code sessions,
and no synthetic example ever enters it. A smoke test with crude hand-written
templates scored **93% cross-validated on synthetic and 55% on the real 40** --
the model had learned the templates, not the concept. Cross-validation on
generated data only proves the generator is self-consistent; the real 40 is the
number that counts.

**The trained head is now the default.** Seeding the generator with real briefs
was the change that made synthetic data transfer; local generation with Ollama
(~800 briefs/hour, against ~65/hour for four throttled hosted free tiers
combined) was what made the corpus big enough to matter.

```
n       cross-val (synthetic)   REAL 40 (held out)
217             78%                    82%
529             80%                    85%
1188            88%                    88%
2299            90%                    92%
2653            90%                    95%   <- plateau
```

Scored on the 40 real briefs, with 3031 synthetic training briefs:

| | real 40 | short briefs | long briefs |
|---|---|---|---|
| shipped router (difficulty score) | 20/40, 50% | 35% | 65% |
| strip + category (hand-tuned) | 27/40, 68% | 55% | 75% |
| **trained head** | **38/40, 95%** | **95%** | **95%** |

Short and long briefs now score identically, which is the whole point: the
length bias is gone, because payload size is uncorrelated with the label in the
training data. Error profile is 2 overspends and **0 underspends** -- nothing is
sent to a model too weak for it, so no retries are wasted.

27 of those 40 seeded the generator, so the honest number is the 13 it never
saw: **12/13**. The single miss is "Summarize what server.py does and where its
error handling is weak", labelled `edit` and predicted `design` -- arguably a
bad label, since "where is it weak" is a judgement call.

An earlier unseeded run is the counter-example worth keeping in mind: it reached
93% cross-validated on synthetic data while stuck at 55% on the real briefs. The
model had learned how four providers write practice questions. Cross-validation
on generated data measures only whether the generator agrees with itself.

```sh
python training/train_head.py --data 'training/seeded/*.jsonl'   # train + score
python training/train_head.py --curve                            # does more data help?
laya-routerd --mode score      # opt out of the head
```

`RouterClient(mode="head"|"score")` overrides per call site; the default defers
to the sidecar, so switching every project over is a daemon restart rather than
a release of each client. A checkout with no `routing_head.joblib` degrades to
the difficulty score instead of failing.

Provider quirks worth knowing, all of which look fatal and are not:

| status | provider | actually means |
|---|---|---|
| 413 Payload Too Large | Groq | tokens-per-minute budget exhausted |
| 524 | Aion (Cloudflare) | slow model outran the edge proxy |
| 404 Not Found | Gemini | model retired; the **body** names the replacement |
| 429 insufficient_quota | OpenAI | no credits; there is no free API tier |

`--list-models` shows what a key can *see*, which is not always what it can *use*
-- Gemini lists `gemini-2.5-flash` and then refuses it for new keys.

### Other limits

- **Tier assignment is not reproducible across runs.** The orchestrator rewords
  each task afresh, so the same user request scored 1.94 (Sonnet) on one run and
  2.06 (Opus) on the next. It also forwards the request verbatim now and then
  (1/15 here), so the input distribution is a mixture.
- **The capture harness is easy to contaminate.** The orchestrator reads agent
  *descriptions* and echoes them into the brief -- a first pass whose no-op agent
  was described as "Records the task. Does not perform it." leaked that wording
  into 4/15 captures and inflated their difficulty. `tests/test_delegated_corpus.py`
  guards against it; the numbers above are from a clean recapture.
- **The 2.0 s client timeout is marginal here.** Delegated briefs are long enough
  that one route breached it against a 115-776 ms norm, which in production is a
  silent fallback to Opus. `eval_delegated.py` scores with a 30 s timeout so a
  timeout is never mistaken for a decision.
- The cost numbers above are one run. They are not a benchmark: output length
  varies per run, so they show routing working, not how much it saves.

## Before you rely on this to save money

A cascade is not automatically cheaper. Two things to measure first:

1. **Prompt caches are model-scoped.** Splitting traffic across three models
   splits the cache, and on repetitive workloads the lost cache hits can cost
   more than the tier savings.
2. **Opus 5 at `low`/`medium` effort** on everything is the simpler baseline, and
   on current models lower effort often matches prior-generation models at high
   effort. Measure that before adding three models' worth of moving parts.

Judge cost per *completed task*, not per request — a cheap call that needs a
retry or a follow-up turn isn't cheap.

## Files

- `claude_router/router.py` — the classifier and cuts. Tunables: `DIFFICULTY_CUTS`,
  `CHEAP_DOMAINS`, `HIGH_STAKES`. Change them here and every project picks it up on
  the next daemon restart — no reinstall anywhere.
- `claude_router/server.py` — the sidecar (`laya-routerd`), stdlib HTTP, threaded
- `claude_router/client.py` — the zero-dependency client your projects import
- `claude_router/agents.py` — subagent routing for the Claude Agent SDK; needs the
  `[agents]` extra
- `tests/test_agents.py` — hook tests against a stub router; no SDK run, no API key
- `eval_delegated.py` — captures real delegated briefs and scores the router on
  them; `delegated_corpus.json` is the cached corpus
- `tests/test_delegated_corpus.py` — corpus integrity guards; no model, no network
- `training/generate_data.py` — synthesise labelled briefs from any OpenAI-compatible
  endpoint; `training/briefs.jsonl` is the accumulated corpus
- `training/train_head.py` — train the head on frozen laya embeddings, score on the real 40
- `claude_router/head.py` — inference for the trained head; `tests/test_head.py` covers it
- `spike_subagents.py` — the runnable spike, with `--dry-run`
- `tests/test_routing.py` — 17 labeled cases, currently 17/17. Re-run after changing any
  cut; also runnable as a script for the calibration table
- `tests/test_client.py` — client + fallback tests; no torch, no network, no API key
- `example.py` — route, then call the chosen model (streaming, refusal-checked)
- `chat.py` — a terminal chat REPL with sticky per-message routing and live cost
  tracking. A demo of routing against a real bill, NOT a Claude Code replacement:
  no file editing, no bash, no tools.
- `launchd/` — login agent plist + installer

```sh
~/.venvs/laya/bin/python -m pytest tests/test_routing.py -m slow   # the labeled cases; needs [server]
curl -s http://127.0.0.1:8129/health                      # routes served, by tier
```

## Why chat.py routes stickily

Prompt caches are per-model, and a chat resends its whole history every turn.
Re-routing each message pays full price for that history instead of the ~0.1x
cache-read rate. Measured on a 10k-token history at Opus rates: **$0.0325 with a
cache hit vs $0.0775 on a miss — a 2.4x penalty**, and it grows with the thread.

So `chat.py` routes the first message, then only escalates. It never drops back
down mid-thread; `/new` starts fresh. A follow-up like "make it faster" reads as
trivial on its own but inherits the hard thread it belongs to — stickiness is
what catches the context a per-message classifier cannot see.
