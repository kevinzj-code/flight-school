"""The deterministic half of a Reader's check ride: the fact swap.

A grounded adapter is only worth having if it answers from the passage in
front of it and not from what its weights remember. The test that separates
the two (the baseline notes "Step 0 of the document
Pigeon"): take an exam answer that quotes a number from its own passage,
change that number in the passage to one the document could never contain,
ask again, and read the answer.

  followed    the changed number appears   → it read the passage
  memorised   the original number appears  → it answered from the weights
  both        both appear                  → ungradeable, counted apart
  neither     neither                      → ungradeable, counted apart

Asked twice, with the adapter and with it switched off, so the adapter's
contribution is separable from the base's — the same paired-control doctrine
as the rest of the ride. Nothing here is a judge's opinion.
"""
from __future__ import annotations

import random
import re
from typing import Any, Sequence

# A number the document could never contain, per original value, so
# "followed" can never be confused with a number the passage already had.
def swap_for(num: str) -> str:
    if "." in num:
        whole, frac = num.split(".", 1)
        return f"{(int(whole) + 5) % 100}.{(int(frac[0]) + 4) % 10}"
    n = int(num)
    return str(n + 37 if n < 50 else n - 34)


# Not part of a longer number: "4,000" must never be matched as "4" or "000",
# because swapping that digit group leaves mangled text. A comma that ends a
# clause ("in 1808, Congress") is not a thousands separator and does not count.
NUM = re.compile(r"(?<![\d.,])\d+(?:\.\d+)?(?![\d])(?!,\d{3})(?!\.\d)")
# The section label a Reader's rows carry: `[§4]`. Its number is never a fact.
CITE = re.compile(r"\[§\d+\]")

# Numbers written as words. The Constitution has almost no digits — "two
# thirds", "thirty five Years", "the Year one thousand eight hundred and
# eight" — so a digits-only probe found no case at all in its exam. A fraction
# is matched before a cardinal, so swapping "two" never turns "two thirds"
# into "thirty nine thirds".
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_W = "|".join(sorted([*_UNITS, *_TENS, *_SCALES], key=len, reverse=True))
_CARDINAL = rf"(?:{_W})(?:(?:\s+|-)(?:and\s+)?(?:{_W}))*"
_DENOM = ("halves|half|thirds|third|fourths|fourth|quarters|quarter|fifths|fifth|sixths|"
          "sixth|sevenths|seventh|eighths|eighth|ninths|ninth|tenths|tenth|elevenths")
WORDS = re.compile(rf"\b(?:(?P<frac>{_CARDINAL}[\s-]+(?:{_DENOM}))|(?P<card>{_CARDINAL}))\b", re.I)
# Fractions the Constitution never states, for a fraction's swap.
_FRACTION_SWAPS = ("four ninths", "five sevenths")


def _norm(text: str) -> str:
    return re.sub(r"[\s-]+", " ", text.strip().lower())


def words_value(text: str) -> int:
    total = current = 0
    for w in _norm(text).split():
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0
    return total + current


def to_words(n: int) -> str:
    """The Constitution's own style: "one thousand seven hundred and seventy four"."""
    if n < 20:
        return next(w for w, v in _UNITS.items() if v == n)
    if n < 100:
        tens = next(w for w, v in _TENS.items() if v == n - n % 10)
        return tens if n % 10 == 0 else f"{tens} {to_words(n % 10)}"
    if n < 1000:
        rest = n % 100
        return f"{to_words(n // 100)} hundred" + (f" and {to_words(rest)}" if rest else "")
    rest = n % 1000
    joiner = " and " if 0 < rest < 100 else " "
    return f"{to_words(n // 1000)} thousand" + (joiner + to_words(rest) if rest else "")


def _forms(num: str) -> tuple[set[str], set[str]]:
    """(digit strings, normalised word phrases) that all mean `num`."""
    if NUM.fullmatch(num):
        return {num}, ({_norm(to_words(int(num)))} if num.isdigit() and int(num) < 1_000_000 else set())
    phrase = _norm(num)
    m = WORDS.fullmatch(phrase)
    if m and m.group("card"):
        return {str(words_value(phrase))}, {phrase}
    return set(), {phrase}


def mentions(text: str, num: str) -> bool:
    """Whether `text` states `num`, as digits or in words. A phrase counts only
    as a whole number — "two" is not in "twenty two" or in "two thirds"."""
    digits, phrases = _forms(num)
    if any(n in digits for n in NUM.findall(text)):
        return True
    return any(_norm(m.group()) in phrases for m in WORDS.finditer(text))


def _split(question: str) -> tuple[str, str]:
    """A Reader row's heading line (`[§9] Article I, Section 9`) and the rest.
    The heading's numbers name the section; they are never the fact asked."""
    if CITE.match(question) and "\n" in question:
        head, body = question.split("\n", 1)
        return head + "\n", body
    return "", question


def probe_number(question: str, answer: str) -> str | None:
    """The number an answer took from its passage, as the passage writes it —
    digits first, longest first, since a distinctive "3.8" is a better probe
    than a "1" that appears everywhere; then words, largest value first — or
    None when the row cannot be graded this way."""
    _, passage = _split(question)
    passage, answer = CITE.sub("", passage), CITE.sub("", answer)
    in_answer = set(NUM.findall(answer))
    in_passage = set(NUM.findall(passage))
    for n in sorted(in_answer & in_passage, key=len, reverse=True):
        if len(n) >= 2 and passage.count(n) <= 4:
            return n
    found = [m for m in WORDS.finditer(passage)]
    counts: dict[str, int] = {}
    for m in found:
        counts[_norm(m.group())] = counts.get(_norm(m.group()), 0) + 1
    def rank(m):
        return (1 if m.group("frac") else 0, words_value(m.group()))
    for m in sorted(found, key=rank, reverse=True):
        phrase = _norm(m.group())
        if m.group("card") and words_value(phrase) < 2:
            continue                                   # "one" is a pronoun as often as a number
        if counts[phrase] <= 4 and mentions(answer, phrase):
            return m.group()
    return None


def swap_in(question: str, num: str) -> tuple[str, str]:
    """The question with `num` changed everywhere in its passage, and the
    changed number as written there."""
    head, body = _split(question)
    if NUM.fullmatch(num):
        swap = swap_for(num)
        return head + NUM.sub(lambda m: swap if m.group() == num else m.group(), body), swap
    phrase = _norm(num)
    m = WORDS.fullmatch(phrase)
    if m and m.group("frac"):
        swap = next(f for f in _FRACTION_SWAPS if f not in _norm(body))
    else:
        n = int(swap_for(str(words_value(phrase))))
        while _norm(to_words(n)) in _norm(body):
            n += 1
        swap = to_words(n)

    def sub(m: re.Match) -> str:
        if _norm(m.group()) != phrase:
            return m.group()
        return swap[0].upper() + swap[1:] if m.group()[0].isupper() else swap
    return head + WORDS.sub(sub, body), swap


def swap_cases(exam_rows: Sequence[dict[str, Any]], *, limit: int = 12, seed: int = 11) -> list[dict[str, Any]]:
    """The exam rows that can be fact-swapped, with the swap made. Refusal
    rows are skipped — their passage does not answer the question, so a number
    in it proves nothing."""
    cases = []
    for r in exam_rows:
        if r.get("refusal"):
            continue
        q, a = str(r.get("question", "")), str(r.get("answer", ""))
        num = probe_number(q, a)
        if not num:
            continue
        altered, swap = swap_in(q, num)
        cases.append({"question": q, "altered": altered, "num": num, "swap": swap, "expected": a})
    random.Random(seed).shuffle(cases)
    return cases[:limit]


def verdict(out: str, num: str, swap: str) -> str:
    has_swap, has_orig = mentions(out, swap), mentions(out, num)
    return ("followed" if has_swap and not has_orig else
            "memorised" if has_orig and not has_swap else
            "both" if has_swap and has_orig else "neither")


def tally(verdicts: Sequence[str]) -> dict[str, int]:
    return {k: sum(1 for v in verdicts if v == k) for k in ("followed", "memorised", "both", "neither")}


def empty() -> dict[str, Any]:
    return {"n": 0, "cases": [], "adapter": tally([]), "control": tally([]),
            "adapter_n": 0, "control_n": 0}


def run_swaps(srv, cases: Sequence[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    """Ask each question as written and with its number changed, with the
    adapter, then without. The two adapter states never interleave (fly.py says
    why: KV reuse across adapter states), but within one every request has the
    same state, so each pass fans out over the server's slots exactly as the
    check ride's own passes do. No system prompt — the training had none.

    A case COUNTS for a side only when that side, asked the question as
    written, states the number. Otherwise "neither" after the swap proves
    nothing: the answer never needed the number, or the probe picked one the
    question was not about. Measured on the Constitution (43 cases): 17 of
    pat-local-draft's 21 "neither" were cases like that.
    """
    if not cases:
        return empty()
    from .fly import _fan_out
    slots = max(1, getattr(srv.spec, "n_parallel", 1))

    def ask_all(lora, key: str) -> list[str]:
        answers: list[str] = [""] * len(cases)

        def ask(i: int) -> tuple[int, str]:
            return i, srv.chat([{"role": "user", "content": cases[i][key]}],
                               max_tokens=max_tokens, temperature=0.0, seed=11,
                               think=False, lora=lora).strip()

        def put(result: tuple[int, str]) -> None:
            answers[result[0]] = result[1]

        _fan_out(ask, range(len(cases)), slots, put)
        return answers

    off = [{"id": 0, "scale": 0.0}]
    plain_adapter, with_adapter = ask_all(None, "question"), ask_all(None, "altered")
    plain_control, without = ask_all(off, "question"), ask_all(off, "altered")
    out = []
    for c, pa, a, pc, b in zip(cases, plain_adapter, with_adapter, plain_control, without):
        out.append({**c,
                    "adapter_plain": pa, "adapter_counts": mentions(pa, c["num"]),
                    "adapter_answer": a, "adapter": verdict(a, c["num"], c["swap"]),
                    "control_plain": pc, "control_counts": mentions(pc, c["num"]),
                    "control_answer": b, "control": verdict(b, c["num"], c["swap"])})
    return {"n": len(out), "cases": out,
            "adapter": tally([x["adapter"] for x in out if x["adapter_counts"]]),
            "control": tally([x["control"] for x in out if x["control_counts"]]),
            "adapter_n": sum(x["adapter_counts"] for x in out),
            "control_n": sum(x["control_counts"] for x in out)}


def source_check(flight, chunks: Sequence[dict] | None) -> dict | None:
    """A Reader's check that needs no judge: for every exam row, does the
    answer come from the same section and sentence as the expected one?
    (pigeon/support.py `same_source`, the rule Pigeon checks a grounded
    Pigeon's examples by.) A right answer in other words passes; one from
    another sentence, or citing a section it was not shown, does not. Marks
    each row and control row with `same_source`, and returns
    {same, n, base_same, base_n}, or None when no row carries its section."""
    from pigeon import runner, support
    if not chunks:
        return None

    def mark(rows) -> tuple[int, int, int]:
        same = sentence = n = 0
        for row in rows:
            split = runner.split_grounded(row.question, list(chunks))
            if split is None:
                continue
            chunk, _ = split
            m = support.same_source(int(chunk["ordinal"]), chunk["text"], row.expected, row.actual)
            row.same_source = m.to_json()
            n += 1
            same += m.same
            sentence += m.sentence
        return same, sentence, n

    same, sentence, n = mark(flight.rows)
    if not n:
        return None
    base_same, base_sentence, base_n = mark(flight.control_rows or [])
    # `sentence` is the one to read beside the untrained model: it never
    # learned to cite, so `same` (which needs the citation, as Pigeon's
    # example check does) would credit the adapter with a habit, not a reading.
    return {"same": same, "sentence": sentence, "n": n,
            "base_same": base_same if base_n else None, "base_sentence": base_sentence if base_n else None,
            "base_n": base_n or None}
