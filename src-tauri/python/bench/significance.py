"""The paired test. Ported from the earlier harness's `score.py:471` and `audit.py:35`.

WHY A PAIRED TEST AND NOT A DIFFERENCE OF TOTALS. Two runs of the same corpus
at different seeds scored 25/26 and 24/26 — a one-case "difference" that is
pure noise. Comparing totals cannot tell that apart from a real fix. What can
is the DISCORDANT pairs: how many cases B fixed that A missed, against how many
it broke. Cases both got right, or both got wrong, carry no information about
which is better and are correctly ignored.

No scipy. The exact two-sided binomial is a handful of integer operations, and
adding a dependency to the training venv for it would be a poor trade.
"""

from __future__ import annotations


def discordant(a_hits, b_hits) -> tuple[int, int]:
    """(fixed, broken) — the only two counts the test reads.

    `fixed` is cases B got right that A got wrong; `broken` the reverse.
    """
    if len(a_hits) != len(b_hits):
        raise ValueError(
            f"paired test needs equal-length results, got {len(a_hits)} and "
            f"{len(b_hits)} — a paired test over unaligned cases is not a test"
        )
    fixed = sum(1 for a, b in zip(a_hits, b_hits) if not a and b)
    broken = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    return fixed, broken


def _comb(n: int, k: int) -> int:
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def mcnemar(a_hits, b_hits) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired boolean outcomes.

    Returns (fixed, broken, p). `p` is the exact binomial two-sided test on
    fixed+broken trials at 0.5.

    No discordant pairs at all means the two runs agreed everywhere, which is
    p = 1.0: no evidence of a difference, rather than evidence of none.
    """
    fixed, broken = discordant(a_hits, b_hits)
    n = fixed + broken
    if n == 0:
        return fixed, broken, 1.0
    k = min(fixed, broken)
    tail = sum(_comb(n, i) for i in range(k + 1))
    p = min(1.0, 2.0 * tail / (2 ** n))
    return fixed, broken, p


def mcnemar_floor(n_discordant: int) -> float:
    """Smallest two-sided p reachable with `n_discordant` pairs all one way.

    This is the number that says whether a bench COULD have proved anything
    before it was run. Six discordant pairs is the first count that can clear
    0.05 — with five, a perfect 5-fixed/0-broken result still only reaches
    0.0625, and reporting it as a win is reporting noise.

    And the discordant pairs available are capped by how many cases the
    BASELINE gets wrong: a bench whose baseline scores 100/100 has zero
    discriminating power however large it is.
    """
    return min(1.0, 2.0 / (2 ** n_discordant)) if n_discordant else 1.0


def verdict(fixed: int, broken: int, p: float, alpha: float = 0.05) -> str:
    """One plain line for a report. Deliberately refuses to say "better" on a
    result that has not cleared the bar — that phrasing is how a noise result
    becomes a shipped claim.
    """
    if fixed == 0 and broken == 0:
        return "no difference — the two runs agreed on every case"
    direction = "fixed" if fixed > broken else "broke"
    n = fixed + broken
    if p < alpha:
        return (f"{fixed} fixed / {broken} broken, p = {p:.4g} — significant "
                f"(the change {direction} more than chance explains)")
    floor = mcnemar_floor(n)
    if floor >= alpha:
        return (f"{fixed} fixed / {broken} broken, p = {p:.4g} — NOT significant, "
                f"and could not have been: {n} discordant pair(s) can reach "
                f"p = {floor:.4g} at best. The bench lacked the power, not the "
                f"change the effect")
    return (f"{fixed} fixed / {broken} broken, p = {p:.4g} — directional, not "
            f"significant. Treat as noise until it repeats")
