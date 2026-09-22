"""A terminal chat REPL that picks a Claude model per message.

    pip install ~/claude-router anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python ~/claude-router/chat.py

This is a chat client, not Claude Code -- no file editing, no bash, no tools.
It exists to show per-message routing against a real conversation and a real bill.

Routing is STICKY on purpose. Prompt caches are per-model: the conversation
history is resent every turn, and re-routing every turn would pay full price for
all of it instead of the ~0.1x cache-read rate. So we route the first message,
then only ESCALATE (never quietly drop back down) unless you start a new thread
with /new. Escalating costs one cache miss and buys a better answer; flip-flopping
costs a cache miss every single turn and buys nothing.

Commands:  /new  /model <tier|auto>  /stats  /quit
"""
import os
import sys

import anthropic
from claude_router import TIERS, RouterClient

TIER_ORDER = ["haiku", "sonnet", "opus"]
CACHE_WRITE, CACHE_READ = 1.25, 0.10      # multipliers on the input rate


class Chat:
    def __init__(self, max_tokens: int = 8000):
        self.router = RouterClient()
        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.pinned = None                 # a tier forced via /model, or None for auto
        self.reset()
        self.spend = 0.0
        self.turns = 0

    def reset(self):
        self.messages = []
        self.tier = None                   # sticky tier for this thread

    def pick_tier(self, prompt: str):
        if self.pinned:
            return self.pinned, "pinned via /model"
        route = self.router.route(prompt)
        if self.tier is None:
            return route.tier, route.reason
        # Sticky: escalate freely, never de-escalate mid-thread (it would cost a
        # full cache miss on the whole history to save a little on one reply).
        if TIER_ORDER.index(route.tier) > TIER_ORDER.index(self.tier):
            return route.tier, "escalated: " + route.reason
        return self.tier, "sticky %s (this turn looked like %s)" % (self.tier, route.tier)

    @staticmethod
    def kwargs_for(tier: str):
        model = TIERS[tier][0]
        if tier == "haiku":                # Haiku 4.5 rejects effort and adaptive thinking
            return {"model": model}
        return {"model": model, "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high" if tier == "opus" else "medium"}}

    def cost(self, tier: str, usage) -> float:
        _, pin, pout = TIERS[tier]
        fresh = getattr(usage, "input_tokens", 0) or 0
        write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        return (fresh * pin + write * pin * CACHE_WRITE + read * pin * CACHE_READ
                + out * pout) / 1e6

    def send(self, prompt: str):
        tier, why = self.pick_tier(prompt)
        switched = self.tier is not None and tier != self.tier
        self.tier = tier
        print("\033[2m[%s] %s%s\033[0m" % (
            TIERS[tier][0], why, "  (cache reset by the switch)" if switched else ""),
            file=sys.stderr)

        self.messages.append({"role": "user", "content": prompt})
        with self.client.messages.stream(
            **self.kwargs_for(tier),
            max_tokens=self.max_tokens,
            # Cache the history prefix; this is what stickiness protects.
            cache_control={"type": "ephemeral"},
            messages=self.messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            msg = stream.get_final_message()
        print()

        if msg.stop_reason == "refusal":
            cat = msg.stop_details.category if msg.stop_details else "unknown"
            print("\033[31m[refused: %s]\033[0m" % cat, file=sys.stderr)
            self.messages.pop()
            return

        self.messages.append({"role": "assistant", "content": msg.content})
        c = self.cost(tier, msg.usage)
        self.spend += c
        self.turns += 1
        u = msg.usage
        print("\033[2m  $%.5f this turn, $%.4f total | in=%s cached=%s out=%s\033[0m" % (
            c, self.spend, u.input_tokens,
            getattr(u, "cache_read_input_tokens", 0) or 0, u.output_tokens), file=sys.stderr)

    def command(self, line: str) -> bool:
        cmd, _, arg = line[1:].partition(" ")
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd == "new":
            self.reset(); print("\033[2mnew thread; routing reset to auto\033[0m", file=sys.stderr)
        elif cmd == "model":
            arg = arg.strip().lower()
            if arg in ("auto", ""):
                self.pinned = None; print("\033[2mrouting: auto\033[0m", file=sys.stderr)
            elif arg in TIERS:
                self.pinned = arg; print("\033[2mpinned to %s\033[0m" % TIERS[arg][0], file=sys.stderr)
            else:
                print("\033[2musage: /model auto|%s\033[0m" % "|".join(TIER_ORDER), file=sys.stderr)
        elif cmd == "stats":
            print("\033[2m%d turns, $%.4f, current tier %s, %d messages in thread\033[0m" % (
                self.turns, self.spend, self.tier or "-", len(self.messages)), file=sys.stderr)
        else:
            print("\033[2mcommands: /new /model <tier|auto> /stats /quit\033[0m", file=sys.stderr)
        return True


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("set ANTHROPIC_API_KEY (or run `ant auth login`)", file=sys.stderr)
    chat = Chat()
    print("\033[2mrouted chat. /new /model /stats /quit\033[0m", file=sys.stderr)
    while True:
        try:
            line = input("\n\033[1m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            if not chat.command(line):
                break
            continue
        try:
            chat.send(line)
        except anthropic.APIStatusError as e:
            print("\033[31mAPI error %s: %s\033[0m" % (e.status_code, e.message), file=sys.stderr)
        except anthropic.APIConnectionError:
            print("\033[31mnetwork error\033[0m", file=sys.stderr)
    print("\n\033[2m%d turns, $%.4f total\033[0m" % (chat.turns, chat.spend), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
