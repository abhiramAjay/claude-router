"""Guards on the delegated-phrasing corpus. No model, no network, no API key.

These check that the corpus is *usable evidence* -- that it is complete, covers
every tier, and is free of the harness's own wording. They deliberately do not
assert a routing score: the router currently gets 3/15 here, and that failure is
the finding (see README, "Delegated phrasing breaks the calibration"), not a
regression to fix by moving a cut.
"""
import json
import os
import re

import pytest

import eval_delegated

LEAKED = re.compile(r"record (this|the) task|do(es)? not perform|RECORDED", re.I)


@pytest.fixture(scope="module")
def corpus():
    if not os.path.exists(eval_delegated.CORPUS):
        pytest.skip("no corpus; run `python eval_delegated.py --capture`")
    return json.load(open(eval_delegated.CORPUS))


def test_corpus_covers_every_case(corpus):
    assert [c["request"] for c in corpus] == [r for r, _ in eval_delegated.CASES]


def test_corpus_covers_every_tier(corpus):
    assert {c["want"] for c in corpus} == {"haiku", "sonnet", "opus"}


def test_no_capture_harness_wording_leaked(corpus):
    # The orchestrator echoes agent *descriptions* into the brief it writes. A
    # first pass whose no-op agent was described as "Records the task. Does not
    # perform it." leaked that into 4/15 captures and inflated their difficulty.
    leaked = [c["request"] for c in corpus if LEAKED.search(c["delegated"])]
    assert not leaked, "harness wording leaked into: %s" % leaked


def test_delegated_prompts_are_mostly_paraphrased(corpus):
    # The premise of the whole exercise: what the router sees is a brief the
    # orchestrator wrote, not the request the user typed. Not always, though --
    # it forwards the request verbatim now and then (1/15 here), which is why
    # this is a majority check rather than "all".
    verbatim = [c["request"] for c in corpus if c["delegated"] == c["request"]]
    assert len(verbatim) < len(corpus) / 2, "mostly verbatim: %s" % verbatim
    ratio = (sum(len(c["delegated"]) for c in corpus)
             / sum(len(c["request"]) for c in corpus))
    assert ratio > 3, "expected delegated briefs to be several times longer, got %.1fx" % ratio
