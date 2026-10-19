"""Flight School's measurement subsystem — ported from an earlier measurement harness.

This package exists so that the discipline survives the merge. Every check in
here was written because something got past the previous ones, and the comments
name the failure rather than describing the code.

The four things it provides, and why each one is not optional:

  significance   A one- or two-case "win" on a small gate is noise. Run-to-run
                 noise was MEASURED at about one case on a 47-case gate
                 (dispatcher-v2 25/26 against the same corpus at a different
                 seed, 24/26, p = 1.0). Raw totals lie at this sample size;
                 the paired test is the only honest read.

  leak_guard     A "held-out" case that is a gate case with the numbers changed
                 is not held out. The near-duplicate check found 12 such rows in
                 one audit, touching 5 of 47 gate cases.

  corpus_audit   A training set skewed enough to teach a false rule. The
                 plotter corpus taught "a table means labels=" from a 28:7
                 imbalance and began inventing labels for every bare table.

  Nothing here talks to a model, a server or the filesystem. It is arithmetic
  and string comparison, so it is fast enough to run on every build and
  testable without any of the above.

Ported from the earlier harness's `score.py`, `audit.py`, `traindata.py`, `plotterdata.py`
and `export.py`. That repo stays a frozen archive; fixes go here.
"""

from .significance import mcnemar, mcnemar_floor, discordant
from .leak import too_close, LEAK_THRESHOLD, find_leaks
from .corpus import (
    label_ratio_problems,
    sequence_length_problems,
    contradiction_problems,
    duplicate_problems,
    subject_variety_problems,
    content_words,
    audit_rows,
    CHARS_PER_TOKEN_DENSE,
    CHARS_PER_TOKEN_PROSE,
    MAX_LABEL_RATIO,
    MAX_SHARE_OF_ONE_LABEL,
)

__all__ = [
    "mcnemar", "mcnemar_floor", "discordant",
    "too_close", "LEAK_THRESHOLD", "find_leaks",
    "label_ratio_problems", "sequence_length_problems",
    "contradiction_problems", "duplicate_problems", "subject_variety_problems", "content_words", "audit_rows",
    "CHARS_PER_TOKEN_DENSE", "CHARS_PER_TOKEN_PROSE",
    "MAX_LABEL_RATIO", "MAX_SHARE_OF_ONE_LABEL",
]
