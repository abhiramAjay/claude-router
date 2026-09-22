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

Measured over the labeled cases in `eval_router.py`:

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
- `example.py` has not been run against the live API — there are no Anthropic
  credentials on this machine. The routing, the eval and the emitted request
  params are all verified; the HTTP call is not.

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
- `tests/test_routing.py` — 17 labeled cases, currently 17/17. Re-run after changing any
  cut; also runnable as a script for the calibration table
- `tests/test_client.py` — client + fallback tests; no torch, no network, no API key
- `example.py` — route, then call the chosen model (streaming, refusal-checked)
- `chat.py` — a terminal chat REPL with sticky per-message routing and live cost
  tracking. A demo of routing against a real bill, NOT a Claude Code replacement:
  no file editing, no bash, no tools.
- `launchd/` — login agent plist + installer

```sh
~/.venvs/laya/bin/python ~/ideaProjects/claude-router/eval_router.py   # needs the [server] extra
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
