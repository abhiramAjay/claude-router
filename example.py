"""End-to-end: route via the sidecar, then call the chosen Claude model.

    pip install /path/to/claude-router anthropic
    python example.py "your prompt here"

Needs credentials: an exported ANTHROPIC_API_KEY, or `ant auth login` (the
zero-arg client picks up the stored profile automatically).
"""
import sys

import anthropic
from claude_router import RouterClient

router = RouterClient()          # localhost:8129, falls back to Opus if the daemon is down
client = anthropic.Anthropic()


def ask(prompt: str, max_tokens: int = 16000) -> str:
    route = router.route(prompt)
    print("-> %s (effort=%s) | %s" % (route.model, route.effort, route.reason), file=sys.stderr)

    # Streaming: max_tokens is large and Opus at xhigh can run long, so this
    # avoids the SDK's HTTP timeout on slow turns.
    with client.messages.stream(
        **route.messages_kwargs(),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        return "[refused: %s]" % (message.stop_details.category if message.stop_details else "unknown")
    return "".join(b.text for b in message.content if b.type == "text")


if __name__ == "__main__":
    print(ask(" ".join(sys.argv[1:]) or "what's the capital of france"))
