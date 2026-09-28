# claude-router

[![tests](https://github.com/abhiramAjay/claude-router/actions/workflows/tests.yml/badge.svg)](https://github.com/abhiramAjay/claude-router/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Claude Code runs every subagent on the same model. claude-router picks the model
per task, using a local classifier, before the subagent starts: Haiku for lookups,
Sonnet for contained edits, Opus for design work.**

The decision takes ~100 ms on your own machine and costs zero tokens. On 40 real
delegated tasks captured from Claude Code sessions, it picks the right tier
**95% of the time** and never once sends a task to a model too weak for it.

It also gives every subagent a **shared memory of the session**, so a subagent
knows what you asked and what earlier subagents found, even if its brief left
that out.

```
you ─▶ Claude Code ─▶ Agent("Find where parse_config is defined") ──┐
                                                                    │  PreToolUse hook
                         laya sidecar (local, ~100 ms, $0) ◀────────┘
                                   │
                        "lookup" ─▶ model: haiku ─▶ subagent runs on Haiku 4.5
```

---

## Why

When Claude Code delegates, the subagent inherits one model for everything. A
"find the file that defines X" task and a "design a distributed rate limiter"
task cost the same per token, and the gap between Haiku and Opus is 5x.

Asking an LLM to choose the model adds a network round trip and a bill to every
delegation. claude-router uses [laya](https://pypi.org/project/laya), a 421M
ModernBERT, as a frozen encoder with a small trained classifier on top. It runs
locally, offline, and fast enough to sit inside a hook.

## Features

- **Per-task model routing in the interactive `claude` CLI**, through a
  `PreToolUse` hook. It runs on your normal Claude Code login, with no API key.
- **A shared context pool for subagents.** User messages, briefs and subagent
  results are recorded per session and appended to each new brief (capped at
  about 3k tokens).
- **A high-stakes guard.** Anything that looks destructive (`drop table`,
  `rm -rf`, `force-push`, production deploys) or medical is always sent to Opus,
  whatever the classifier says.
- **Fails open.** If the sidecar is down or slow, the hook steps aside and Claude
  Code uses its own default. Nothing breaks.
- **A zero-dependency Python client** for routing your own Anthropic API calls,
  plus an integration for the Claude Agent SDK.
- **A reproducible training pipeline.** Synthetic data from any OpenAI-compatible
  endpoint (Ollama works) is scored against real captured briefs.

## Results

Tier accuracy on 40 briefs captured from real Claude Code sessions. None of them
appear in the training data:

| Router | Accuracy | Short briefs | Long briefs |
|---|---|---|---|
| laya difficulty score (baseline) | 50% | 35% | 65% |
| Prompt stripping + hand-tuned categories | 68% | 55% | 75% |
| **Trained head (default)** | **95%** | **95%** | **95%** |

Both errors are overspends, not underspends, so no task gets sent to a model too
weak for it. [docs/RESEARCH.md](docs/RESEARCH.md) explains how this was measured,
including why the baseline turned out to be mostly measuring prompt length.

| Tier | Model | Chosen for |
|---|---|---|
| `lookup` | `claude-haiku-4-5` | find, list, read, report |
| `edit` | `claude-sonnet-5` | contained changes, fixes, tests |
| `design` | `claude-opus-5` | architecture, open-ended design, and anything the high-stakes guard catches |

---

## Setup

**Requirements:** Python 3.10+, Claude Code 2.1+, about 2 GB of disk for the
model. The login agent is for macOS; on Linux, run the sidecar by hand or under
systemd (see below).

### 1. Clone and install the sidecar

The classifier lives in one background process (the *sidecar*), so the model is
loaded once and everything else stays lightweight.

```sh
git clone https://github.com/abhiramAjay/claude-router.git
cd claude-router

python3 -m venv ~/.venvs/laya
~/.venvs/laya/bin/pip install '.[server,head]'
```

### 2. Start it

**macOS, starting at login and restarting if it dies:**

```sh
sh launchd/install.sh
```

**Anywhere, in the foreground:**

```sh
~/.venvs/laya/bin/laya-routerd --verbose
```

The first start downloads the model (~800 MB, one time). After that it loads in
about 20 s. Check it:

```sh
curl -s http://127.0.0.1:8129/health     # {"ok": true, "loaded": true, ...}
```

The sidecar binds to `127.0.0.1` only and has no authentication, so don't expose
the port.

### 3. Turn it on in Claude Code

**Just this repo:** nothing to do. `.claude/settings.json` already enables the
hooks, so run `claude` from the repo root.

**Every project:** add these hooks to `~/.claude/settings.json`, with the path
pointing at your clone. The hook itself is standard library only, so any
`python3` can run it.

```json
{
  "hooks": {
    "PreToolUse":       [{ "matcher": "Agent|Task", "hooks": [{ "type": "command", "timeout": 10,
                           "command": "PYTHONPATH=/path/to/claude-router python3 -m claude_router.cc_hook" }] }],
    "SessionStart":     [{ "hooks": [{ "type": "command", "timeout": 10,
                           "command": "PYTHONPATH=/path/to/claude-router python3 -m claude_router.cc_hook --warm" }] }],
    "UserPromptSubmit": [{ "hooks": [{ "type": "command", "timeout": 10,
                           "command": "PYTHONPATH=/path/to/claude-router python3 -m claude_router.cc_hook --record" }] }],
    "PostToolUse":      [{ "matcher": "Agent|Task", "hooks": [{ "type": "command", "timeout": 10,
                           "command": "PYTHONPATH=/path/to/claude-router python3 -m claude_router.cc_hook --record" }] }],
    "SubagentStop":     [{ "hooks": [{ "type": "command", "timeout": 10,
                           "command": "PYTHONPATH=/path/to/claude-router python3 -m claude_router.cc_hook --record" }] }]
  }
}
```

| Hook | What it does |
|---|---|
| `PreToolUse` | routes the subagent, then appends the shared context |
| `SessionStart` | warms the model so the first delegation isn't slow, and tells the main session that the pool exists |
| `UserPromptSubmit`, `PostToolUse`, `SubagentStop` | record messages and results into the pool |

Only routing needs the `PreToolUse` hook. The other four power the shared
context pool.

### 4. Watch it work

Ask Claude Code something it will delegate, then look at the decision log:

```sh
python3 examples/watch_routes.py
```

```
14:02:11  haiku   Find sidecar default port             head: haiku p=1.00
14:02:11  sonnet  Add --timeout flag to laya-routerd    head: sonnet p=0.92  +0.4k shared context
14:02:12  opus    Design multi-machine routing          head: opus p=0.99  +0.4k shared context
```

Each line shows the task, the model chosen and the classifier's confidence. The
raw log is `~/Library/Logs/laya-cc-hook.jsonl`. For a session that
delegates everything, try the included coordinator agent:

```sh
mkdir -p ~/.claude/agents && cp examples/agents/orchestrator.md ~/.claude/agents/
claude --agent orchestrator
```

### Linux (systemd)

```ini
# ~/.config/systemd/user/laya-routerd.service
[Unit]
Description=claude-router laya sidecar

[Service]
ExecStart=%h/.venvs/laya/bin/laya-routerd --host 127.0.0.1 --port 8129
Environment=TOKENIZERS_PARALLELISM=false
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now laya-routerd
```

### Uninstall

```sh
sh launchd/uninstall.sh          # macOS login agent
rm -rf ~/.claude/laya-pool       # shared context pools
```

Then remove the hooks from your settings.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_ROUTER_URL` | `http://127.0.0.1:8129` | where the sidecar listens |
| `CLAUDE_ROUTER_FALLBACK` | `claude-opus-5` | model the Python client returns when the sidecar is unreachable |
| `CLAUDE_ROUTER_POOL` | on | set to `0` to turn off the shared context pool |
| `CLAUDE_ROUTER_POOL_DIR` | `~/.claude/laya-pool` | where session pools are stored |
| `CLAUDE_ROUTER_HOOK_LOG` | `~/Library/Logs/laya-cc-hook.jsonl` | the hook's decision log |

Sidecar flags: `laya-routerd --host 127.0.0.1 --port 8129 --mode head|score --verbose`.
`--mode score` switches back to the untrained difficulty score.

To change routing behaviour, edit `claude_router/router.py`: `DIFFICULTY_CUTS`,
`CHEAP_DOMAINS`, and the `HIGH_STAKES` regex. Restart the sidecar to pick up
changes; no client needs reinstalling.

## Privacy

Everything runs locally. The classifier makes no network calls once the model is
downloaded (the launchd agent sets `HF_HUB_OFFLINE=1`). The context pool holds your
messages and subagent output, and the decision log holds task descriptions. Both
stay on your disk.
Pools older than 7 days are deleted automatically.

---

## Using it from your own code

### Anthropic API

Install the client into any project. It is standard library only: no torch, no
dependencies, 13 MB.

```sh
pip install /path/to/claude-router
```

```python
import anthropic
from claude_router import RouterClient

router = RouterClient()
route = router.route(prompt)          # ~70 ms over HTTP to the sidecar
print(route.model, route.reason)

anthropic.Anthropic().messages.create(
    **route.messages_kwargs(), max_tokens=16000,
    messages=[{"role": "user", "content": prompt}],
)
```

`messages_kwargs()` emits the right parameters for each model. Haiku gets no
`thinking` or `effort`, since it rejects both. Sonnet and Opus get adaptive
thinking plus an effort level. If the sidecar is down, `route()` returns the
fallback model. Use `RouterClient(on_error="raise")` to raise an error instead.

`example.py` is a minimal script that routes a prompt and calls the chosen
model. `chat.py` is a terminal chat with per-thread routing and a live cost
counter.

### Claude Agent SDK

```sh
pip install '/path/to/claude-router[agents]'
```

```python
from claude_agent_sdk import query
from claude_router.agents import DELEGATION_INSTRUCTION, TaskRouter, routing_options

router = TaskRouter()
options = routing_options(router=router, system_prompt={
    "type": "preset", "preset": "claude_code", "append": DELEGATION_INSTRUCTION})
async for message in query(prompt="...", options=options):
    ...
print(router.summary())               # what was routed where, and why
```

`python spike_subagents.py --dry-run` prints the routing table without calling
the API.

---

## How it works

1. **The hook intercepts delegation.** Claude Code calls the `Agent` tool with a
   brief. A `PreToolUse` hook reads the brief before the subagent spawns.
2. **The sidecar classifies it.** laya embeds the raw brief (~38 ms). A logistic
   regression head trained on those embeddings labels it `lookup`, `edit` or
   `design`, and the high-stakes regex can raise the tier to Opus.
3. **The hook sets `model`.** It returns `updatedInput` with `model: haiku |
   sonnet | opus`. It leaves `subagent_type` alone, so Explore, Plan and custom
   agents keep their tools.
4. **The pool is appended after routing**, so the classifier judges the task
   itself rather than the pasted context.

Why it routes subagents rather than switching the main conversation's model:
prompt caches are tied to a single model, so switching mid-thread gives up the
cache. That costs about 2.4x on a 10k-token history. A subagent starts with a
fresh context, so choosing its model costs nothing.

## Limitations

- **Only delegated work is routed.** The main conversation stays on whatever
  model you started it with.
- **The hook sets the model but not effort.** The CLI's Agent tool takes no
  per-call effort parameter. The Agent SDK integration does set effort.
- **The high-stakes guard is a keyword list.** It is precise, but it knows
  nothing about your domain, so extend `HIGH_STAKES` for yours.
- **The evaluation set is 40 tasks.** 95% on 40 is a strong signal but not a
  benchmark. Only 13 of the 40 were never used to seed the generator, and the
  head gets 12 of those 13 right.

## Retraining the head

```sh
# generate labelled briefs from any OpenAI-compatible endpoint (local Ollama works)
python training/generate_data.py --base-url http://127.0.0.1:11434/v1 --api-key ollama \
    --model qwen2.5-coder:7b --seed-from-real --n 400 --out training/seeded/briefs-ollama.jsonl

# train on everything, then score against the 40 real briefs
python training/train_head.py --data 'training/seeded/*.jsonl'
```

Payload size is sampled independently of the label, so the head can't learn
"long brief means hard task". [docs/RESEARCH.md](docs/RESEARCH.md) explains why
this matters and covers the provider quirks.

## Development

```sh
pip install -e '.[dev,agents]'   # the Agent SDK is needed by tests/test_agents.py
pytest -m 'not slow'               # client, hook, pool, corpus: no model, no network
pytest -m slow                     # labelled routing cases; needs '.[server,head]'
```

```
claude_router/
  cc_hook.py     Claude Code hook: route, record, warm
  pool.py        shared per-session context pool
  client.py      zero-dependency HTTP client
  server.py      the sidecar (laya-routerd)
  router.py      difficulty-score router, tiers, high-stakes guard
  head.py        trained head inference
  agents.py      Claude Agent SDK integration
training/        synthetic data generation and head training
eval_delegated.py, delegated_corpus.json   capture and score real delegated briefs
launchd/         macOS login agent
examples/agents/ the orchestrator agent
docs/RESEARCH.md measurements, dead ends, and design decisions
```

## License

[MIT](LICENSE)
