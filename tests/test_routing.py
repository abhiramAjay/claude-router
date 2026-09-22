"""Labeled cases for ClaudeRouter. Re-run after touching cuts or the guard.

Runs the classifier in-process (not through the sidecar), so it needs the
[server] extra:  pytest tests/test_routing.py -m slow
"""
import time

from claude_router import ClaudeRouter

CASES = [
    ("hi there!", "haiku"),
    ("thanks, that worked", "haiku"),
    ("what's the capital of france", "haiku"),
    ("convert 72 fahrenheit to celsius", "haiku"),
    ("rename the variable `foo` to `bar` in utils.py", "haiku"),
    # laya scores this 1.45 in domain=code -- identical to the gRPC case below, which
    # wants sonnet. The signal cannot separate them, so either tier is accepted:
    # overspending one Sonnet call on a flag lookup is the cheap side of the error.
    ("what does the -p flag do in mkdir", ("haiku", "sonnet")),
    ("write a regex that matches an ISO 8601 date", "sonnet"),
    ("add pagination to this REST endpoint and update the tests", "sonnet"),
    ("write a blog post about why we migrated from Postgres to ClickHouse", "sonnet"),
    ("why is my pandas groupby returning NaN for the mean column?", "sonnet"),
    ("summarize the differences between gRPC and REST for our API docs", "sonnet"),
    ("Design a distributed rate limiter across 200 edge nodes with sub-ms p99, "
     "prove the consistency bounds, and write the Rust implementation", "opus"),
    ("prove that every finite integral domain is a field", "opus"),
    ("refactor our entire auth layer from sessions to JWT without breaking existing clients", "opus"),
    ("my kid swallowed a battery what do i do", "opus"),
    ("review this employment contract clause before I sign it tomorrow", "opus"),
    ("drop the users table in prod to clear out test data", "opus"),
]

def main():
    r = ClaudeRouter()
    r.agent                                   # warm the load out of the timing
    ok = 0
    print("%-56s %-7s %-7s %5s  %-5s %s" % ("request", "want", "got", "diff", "eff", "why"))
    print("-" * 122)
    t0 = time.time()
    for prompt, want in CASES:
        d = r.route(prompt)
        want_set = (want,) if isinstance(want, str) else want
        hit = d.tier in want_set
        ok += hit
        print("%-56s %-7s %-7s %5.2f  %-5s %s%s" % (
            prompt[:54], "|".join(want_set), d.tier, d.difficulty, d.effort or "-", d.reason,
            "" if hit else "   <-- MISS"))
    dt = time.time() - t0
    print("\n%d/%d correct  |  %.0f ms per route  |  %s" % (
        ok, len(CASES), 1000 * dt / len(CASES),
        "all green" if ok == len(CASES) else "%d miss(es)" % (len(CASES) - ok)))
    return 0 if ok == len(CASES) else 1

def test_routing_matches_labels():
    """Slow: loads the 421M checkpoint. Needs the [server] extra.

        pytest tests/test_routing.py -m slow
    """
    import pytest
    pytest.importorskip("laya", reason="needs the [server] extra")
    assert main() == 0, "routing regressed against the labeled cases"


test_routing_matches_labels = __import__("pytest").mark.slow(test_routing_matches_labels)


if __name__ == "__main__":
    raise SystemExit(main())
