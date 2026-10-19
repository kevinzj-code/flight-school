"""The corpus audit — the structural checks, run before every retrain.

Every check here exists because something got past the previous ones. The
comments name the failure, with its measured numbers, because a threshold
without its story gets "tidied" by the next person to read it.

Ported from the earlier harness's `traindata.py:636` and `plotterdata.py:541`, generalized
off those two corpora's specific shapes so it can run on any Flight School
training set: a row is `(input, label, category)` where `label` is whatever the
model must produce and `category` is the family the row belongs to (or `""`).
"""

from __future__ import annotations

from collections import Counter

from .leak import find_leaks, LEAK_THRESHOLD

# --------------------------------------------------------------------------
# Thresholds. Each is a measurement; the comment is the receipt.

#: Two halves of a contrast may not outnumber each other by more than this.
#: MEASURED: the plotter corpus ran 28 labelled table rows against 7 bare, and
#: the adapter learned "a table means labels=" so firmly it INVENTED labels for
#: every bare table — `Jan,Feb,Mar,Apr,May` against a list of rainfall stations.
#: Every figure correct, every label fabricated. Balancing the halves 28:28
#: fixed it (bare 10/28 -> 18/28).
MAX_LABEL_RATIO = 1.6

#: No single label may take more than this share of the corpus. A contrast
#: taught this hard is how one pilot revision lost the gate, and a refusal
#: taught this hard declines work it could have done.
MAX_SHARE_OF_ONE_LABEL = 0.45

#: Number-dense text tokenizes at 1.84 chars/token with the real Qwen2.5
#: tokenizer — `84.39` is four tokens, and a table is mostly that. MEASURED.
CHARS_PER_TOKEN_DENSE = 1.84

#: Prose runs about 3.5. Using THIS ratio on a dense row put a 924-token table
#: at 527 and hid the problem completely, which is why the dense ratio is the
#: default and this one has to be asked for.
CHARS_PER_TOKEN_PROSE = 3.5


def _is_dense(text: str) -> bool:
    """Digits and punctuation over a fifth of the characters means the prose
    ratio will underestimate it. Deliberately crude: the cost of guessing
    "dense" wrongly is a slightly conservative length estimate, and the cost of
    guessing "prose" wrongly is a silently truncated training row.
    """
    if not text:
        return False
    dense = sum(1 for c in text if c.isdigit() or c in ",.|:;-%$")
    return dense / len(text) > 0.20


def estimate_tokens(text: str) -> float:
    """Token estimate that does not lie about number-dense rows."""
    ratio = CHARS_PER_TOKEN_DENSE if _is_dense(text) else CHARS_PER_TOKEN_PROSE
    return len(text) / ratio


# --------------------------------------------------------------------------
# the checks


def label_ratio_problems(rows, max_ratio: float = MAX_LABEL_RATIO,
                         max_share: float = MAX_SHARE_OF_ONE_LABEL):
    """Balance, two ways: no label may dominate the corpus, and no two labels
    may be lopsided enough to teach a false rule.
    """
    problems = []
    labels = [label for _inp, label, _cat in rows]
    if not labels:
        return ["the corpus is empty"]

    counts = Counter(labels)

    # BALANCE IS ONLY A QUESTION FOR A FIXED ANSWER SET. A behavior cartridge's
    # answers are free text — 147 rows with 147 distinct answers — and "the most
    # common answer appears twice" is not a finding. Running the ratio check
    # there would fire on almost every build, and a warning that always fires is
    # a warning nobody reads.
    #
    # The rule: if the answers are mostly distinct, they are not labels, and
    # this check does not apply. A classifier or a router has a handful of
    # labels over hundreds of rows and is nowhere near this line.
    if len(counts) > max(4, len(labels) * 0.5):
        return problems

    # The share ceiling applies only from THREE labels up. With two, a
    # perfectly balanced corpus is 50/50 and would trip a 45% ceiling every
    # time — the check would fire hardest on exactly the corpus it wants. Two
    # labels are a contrast, and the ratio check below is the right tool for
    # them. The earlier harness ran these as two separate checks over two differently
    # shaped corpora and so never met this case; merging them here does.
    if len(counts) >= 3:
        top, n = counts.most_common(1)[0]
        if n > len(labels) * max_share:
            problems.append(
                f"{top!r} is {n}/{len(labels)} rows ({n / len(labels):.0%}) — past "
                f"the {max_share:.0%} share limit. A label taught this hard gets "
                f"applied to work it does not fit"
            )

    # The ratio between the two most common labels. Checked as a RATIO rather
    # than a count because the failure was relative: 7 bare rows were plenty on
    # their own and useless against 28 labelled ones.
    if len(counts) >= 2:
        (top_l, top_n), (second_l, second_n) = counts.most_common(2)
        if second_n and top_n / second_n > max_ratio:
            problems.append(
                f"{top_l!r} outnumbers {second_l!r} {top_n}:{second_n} "
                f"({top_n / second_n:.1f}:1, past {max_ratio}:1) — at 28:7 the "
                f"adapter invented labels for every bare table. Two halves of a "
                f"contrast have to be balanced to work as one"
            )
    return problems


def sequence_length_problems(rows, seq_limit: int, system_prompt_chars: int = 0,
                             headroom: float = 0.95):
    """Rows that would be TRUNCATED rather than rejected by the trainer.

    This is the quiet one. mlx-lm does not refuse an over-long example; it cuts
    it. The answer is the TAIL of the sequence, so an over-long row trains the
    model to stop mid-answer — which is indistinguishable from the looping
    failure the corpus may have been built to fix in the first place.
    """
    problems = []
    for inp, label, cat in rows:
        est = estimate_tokens(str(inp)) + estimate_tokens(str(label))
        est += system_prompt_chars / CHARS_PER_TOKEN_PROSE
        est += 12  # chat-template scaffolding
        if est > seq_limit * headroom:
            where = f"{cat}: " if cat else ""
            problems.append(
                f"{where}~{est:.0f} tokens is at or past the {seq_limit}-token "
                f"ceiling — the ANSWER is what gets cut, so this row would teach "
                f"a truncated answer: {str(inp)[:60]!r}"
            )
    return problems


def contradiction_problems(rows, keys=None):
    """One input must not carry two different labels.

    A corpus that answers the same question two ways teaches the model that
    both are acceptable, and then it picks whichever it likes on the day.

    `keys` is what counts as "the same input", one per row. It exists because
    the question text is NOT the identity on every corpus: a vision corpus
    reuses one phrasing ("What breed is this?") across every image on purpose,
    so keying on the question alone reports the entire corpus as contradictory.
    Passing image-aware keys does not disable the check there — it makes it ask
    the question that actually matters, which is whether the same IMAGE ever
    carries two different labels.

    Omitted, the input is the key, which is right for text.
    """
    rows = list(rows)
    if keys is None:
        keys = [str(inp) for inp, _label, _cat in rows]
    if len(keys) != len(rows):
        raise ValueError("one key per row, or none at all")

    problems = []
    seen: dict[str, set] = {}
    shown: dict[str, str] = {}
    for (inp, label, _cat), key in zip(rows, keys):
        k = str(key).strip().lower()
        seen.setdefault(k, set()).add(str(label))
        shown.setdefault(k, str(inp))
    for k, labels in seen.items():
        if len(labels) > 1:
            problems.append(
                f"contradictory labels for {shown[k][:60]!r}: {sorted(labels)}"
            )
    return problems


def duplicate_problems(rows, guard, threshold: float = LEAK_THRESHOLD):
    """Training rows that are near-duplicates of the held-out set.

    A gate case with the numbers changed is not a holdout, and a bench the
    model trained on measures nothing.
    """
    inputs = [str(inp) for inp, _label, _cat in rows]
    return [
        f"NEAR A HELD-OUT CASE ({ratio:.2f}): {msg[:60]!r} against {g[:60]!r}"
        for msg, g, ratio in find_leaks(inputs, guard, threshold)
    ]


# --------------------------------------------------------------------------
# subject variety — the failure every diagnosed one had in common

#: Words that carry no subject. Short and boring on purpose; the check is
#: about content words that recur, not grammar.
_STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been
being this that these those it its it's i you he she we they me him her us them my your his our their
what which who whom whose when where why how do does did done have has had having can could should would
will shall may might must not no nor so than too very just also only ever never here there now then once
into onto over under again further about above below between through during before after out off up down
one two three four five six seven eight nine ten first second last next new old same other another any
all each every some most more less much many few own such please thank thanks hi hello""".split())

#: No single content word may sit in more than this share of the rows.
#: MEASURED: the earlier harness's plotter corpus taught "every diagram is a business
#: process" — forty-one rows, one domain — and scored 0/7 then 1/7 on real
#: requests, below the untrained model. Temperature alone was measured to add
#: no new failure coverage; the SUBJECT has to vary.
MAX_SUBJECT_SHARE = 0.6


def content_words(text: str) -> set[str]:
    out = set()
    for raw in str(text).lower().split():
        w = "".join(ch for ch in raw if ch.isalpha())
        if len(w) > 3 and w not in _STOP:
            out.add(w)
    return out


def subject_variety_problems(rows, job_words=(), max_share: float = MAX_SUBJECT_SHARE,
                             min_rows: int = 8):
    """A content word that appears in more than half the inputs is a subject
    the model will learn as part of the job.

    `job_words` are the words that are ALLOWED to be everywhere because they
    are the job — "ticket" in a ticket-subject corpus, "breed" in a dog
    sorter. They come from the plan's own description, so the person does not
    have to think about them — and they are not optional in practice: without
    them a corpus whose every row says "draw" is reported for saying "draw".
    The check cannot tell the job's vocabulary from a leaked subject on its
    own; the plan can. Fewer than `min_rows` rows is too few to call anything
    a pattern.
    """
    rows = list(rows)
    if len(rows) < min_rows:
        return []
    allowed = set()
    for jw in job_words:
        allowed |= content_words(jw)
    df: dict[str, int] = {}
    for inp, _label, _cat in rows:
        for w in content_words(inp) - allowed:
            df[w] = df.get(w, 0) + 1
    problems = []
    for w, n in sorted(df.items(), key=lambda kv: -kv[1]):
        share = n / len(rows)
        if share <= max_share:
            break
        problems.append(
            f"one subject dominates: {w!r} is in {n}/{len(rows)} inputs ({share:.0%}). "
            f"A practice set that always says {w!r} teaches that the job is about {w!r} — "
            f"a plotter corpus once taught 'every diagram is a business process' this way "
            f"and scored below the untrained model on real requests"
        )
    return problems


def audit_rows(rows, *, guard=None, seq_limit: int | None = None,
               system_prompt_chars: int = 0, keys=None, job_words=()):
    """Every structural check, in one call. Returns (problems, balance).

    `rows` is an iterable of `(input, label, category)`. `keys` optionally says
    what counts as "the same input" — needed wherever the input text is
    deliberately reused, as a vision corpus reuses one phrasing across every
    image. `guard` is the held-out set. `seq_limit` is the trainer's max sequence length; pass it and
    the truncation guard runs, omit it and it is skipped rather than guessed —
    a guessed ceiling produces confident nonsense.

    An empty problems list is NOT a promise the corpus is good. It is a promise
    that the failures already paid for are absent from it.
    """
    rows = [tuple(r) for r in rows]
    problems: list[str] = []
    problems += contradiction_problems(rows, keys)
    problems += label_ratio_problems(rows)
    problems += subject_variety_problems(rows, job_words)
    if guard:
        problems += duplicate_problems(rows, guard)
    if seq_limit:
        problems += sequence_length_problems(rows, seq_limit, system_prompt_chars)
    balance = dict(Counter(label for _inp, label, _cat in rows))
    return problems, balance
