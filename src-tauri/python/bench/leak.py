"""The leak guard. ONE copy, replacing three.

The earlier harness carried this same function three times — `traindata.py:59`,
`plotterdata.py:48` and a variant at `export.py:100` — all
`difflib.SequenceMatcher(...).ratio() > 0.82`. Three copies of a threshold is
three chances for them to drift apart, and the number is a measurement, not a
preference: at 0.82 the check found 12 rows that were gate cases with the
numbers changed, touching 5 of one gate's 47 cases.

WHAT THIS CATCHES, AND WHAT IT DOES NOT. It catches a training row that is a
rephrasing of a held-out case. It does NOT catch a training row that shares the
held-out case's SUBJECT while wording it completely differently — and that is
the failure mode that actually cost a training run (see `corpus.py`). Phrasing
similarity is a floor, not the whole of "held out."
"""

from __future__ import annotations

import difflib

#: Measured, not chosen. Lowering it produces false positives on ordinary short
#: questions ("add 5 and 7" against "add 6 and 8"); raising it let the twelve
#: numbers-changed rows through.
LEAK_THRESHOLD = 0.82


def _ratio_lower(a_lower: str, b_lower: str) -> float:
    """Similarity of two already-lowercased strings, so callers that compare
    one string against many guard strings don't re-lowercase either side on
    every pair."""
    return difflib.SequenceMatcher(None, a_lower, b_lower).ratio()


def too_close(msg: str, guard, threshold: float = LEAK_THRESHOLD) -> bool:
    """True if `msg` is a near-duplicate of anything in `guard`.

    `guard` is the held-out set: every message the corpus must not have seen.
    """
    msg_lower = (msg or "").lower()
    return any(_ratio_lower(msg_lower, (g or "").lower()) > threshold for g in guard)


def too_close_any_lower(msg_lower: str, guard_lower, threshold: float = LEAK_THRESHOLD) -> bool:
    """Same check as `too_close`, for a caller that already has both sides
    lowercased — e.g. clustering outputs one at a time against a `seen` list
    that only grows, where lowercasing `seen` fresh on every new item would be
    the same string re-lowered on every prior comparison."""
    return any(_ratio_lower(msg_lower, g_lower) > threshold for g_lower in guard_lower)


def find_leaks(rows, guard, threshold: float = LEAK_THRESHOLD):
    """Every (row_message, guard_message, ratio) that trips the guard.

    Returns the pairs rather than a boolean, because "the corpus leaks" is not
    actionable and "these four rows are these four gate cases with the numbers
    changed" is. Sorted worst-first so a long list still leads with the clearest
    offender.
    """
    hits = []
    guard_lower = [(g, (g or "").lower()) for g in guard]
    for msg in rows:
        msg_lower = (msg or "").lower()
        for g, g_lower in guard_lower:
            r = _ratio_lower(msg_lower, g_lower)
            if r > threshold:
                hits.append((msg, g, r))
    hits.sort(key=lambda h: h[2], reverse=True)
    return hits
