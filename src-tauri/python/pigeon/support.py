"""Where a grounded answer came from: the sentence it leans on, and whether two
answers lean on the same one.

A line-for-line port of Pigeon's highlighter (`material::support_scored`,
`declines`, `same_source` in Pigeon's src-tauri/src/pigeon/
material.rs), so a check ride here and the example check in Pigeon pick the
same sentence for the same answer. `bench/test_support.py` holds them to it on
the fixture Pigeon's own tests use. Change one, change both.

Spans here are character ranges of the section, not Pigeon's UTF-16 offsets;
only equality between two answers' spans is ever used.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

STOP = {"the", "and", "for", "that", "with", "this", "shall", "from", "into", "upon", "than", "them", "they",
        "their", "there", "which", "what", "who", "whom", "when", "where", "how", "does", "did", "have", "has", "had", "been",
        "being", "are", "was", "were", "not", "nor", "but", "any", "all", "such", "may", "can", "each", "one", "two", "under",
        "over", "other"}
STOP_TAIL = {"out", "its", "his", "her"}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
            "nineteenth", "twentieth"]
UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ALSO_FLOOR = 3
SAYS_NOT = ["does not say", "doesn't say", "does not mention", "doesn't mention", "does not state",
            "doesn't state", "does not specify", "doesn't specify", "does not contain", "doesn't contain",
            "not mentioned", "not stated", "no information", "not in the passage", "cannot answer", "can't answer"]

_SPLIT = re.compile(r"[^a-z0-9-]+")
_CITATION = re.compile(r"\[§([0-9]+)\]")
_WS = " \t\n\r\x0c"


def _roman(n: int) -> str:
    out = ""
    for v, s in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out += s
            n -= v
    return out


def _ordinal_number(w: str) -> int | None:
    if w in ORDINALS:
        return ORDINALS.index(w) + 1
    if not w.startswith("twenty"):
        return None
    rest = w[len("twenty"):]
    rest = rest[1:] if rest.startswith("-") else rest
    return 21 + ORDINALS[:7].index(rest) if rest in ORDINALS[:7] else None


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def support_tokens(text: str) -> list[str]:
    out: list[str] = []

    def keep(w: str):
        n = _ordinal_number(w)
        if n is not None:
            out.append(_roman(n))
        elif len(w) >= 3 and w not in STOP and w not in STOP_TAIL:
            out.append(_stem(w))

    for raw in _SPLIT.split(text.lower()):
        if not raw:
            continue
        joined = raw.replace("-", "")
        n = _ordinal_number(raw)
        if n is None:
            n = _ordinal_number(joined)
        if n is not None:
            out.append(_roman(n))
            continue
        for part in (p for p in raw.split("-") if p):
            if part.isascii() and part.isdigit() and len(part) <= 2:
                v = int(part)
                if v < 20:
                    keep(UNITS[v])
                else:
                    keep(TENS[v // 10])
                    if v % 10:
                        keep(UNITS[v % 10])
            else:
                keep(part)
    return out


def strip_citations(answer: str) -> str:
    out, rest = [], answer
    while (i := rest.find("[§")) >= 0:
        out.append(rest[:i])
        j = rest.find("]", i)
        rest = rest[j + 1:] if j >= 0 else ""
    out.append(rest)
    return "".join(out)


def cited(answer: str) -> list[int]:
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def declines(answer: str) -> bool:
    a = strip_citations(answer).lower().replace("’", "'")
    return any(p in a for p in SAYS_NOT)


def sentences(text: str) -> list[tuple[int, int]]:
    """Character ranges of the sentences, trimmed; a sentence ends at . ; ! ?
    before whitespace, or at a blank line."""
    out: list[tuple[int, int]] = []

    def push(a: int, b: int):
        s = text[a:b]
        lead, trail = len(s) - len(s.lstrip()), len(s) - len(s.rstrip())
        if a + lead < b - trail:
            out.append((a + lead, b - trail))

    start = 0
    for i, c in enumerate(text):
        nxt = text[i + 1] if i + 1 < len(text) else ""
        ends = c in ".;!?" and nxt != "" and nxt in _WS
        para = c == "\n" and nxt == "\n"
        if ends or para:
            end = i + 1 if ends else i
            push(start, end)
            start = end
    push(start, len(text))
    return out


def support_scored(section: str, answer: str):
    """(sentence span, second-clause span or None, shared words), or None."""
    want: list[str] = []
    for w in support_tokens(strip_citations(answer)):
        if w not in want:
            want.append(w)
    if not want:
        return None
    floor = 1 if len(want) <= 2 else 2
    spans = sentences(section)
    words = [set(support_tokens(section[a:b])) for a, b in spans]
    n = float(len(words))

    def rarity(w):
        f = float(sum(1 for s in words if w in s))
        return math.log(1.0 + (n - f + 0.5) / (f + 0.5))

    def score(i, among):
        shared = [w for w in among if w in words[i]]
        total = 0.0
        for w in shared:
            total += rarity(w)
        return len(shared), total

    def better(a, b):
        return b is None or a[0] > b[1][0] or (a[0] == b[1][0] and a[1] > b[1][1])

    best = None
    for i in range(len(spans)):
        s = score(i, want)
        if s[0] >= floor and better(s, best):
            best = (i, s)
    if best is None:
        return None
    first, (hits, _) = best
    left = [w for w in want if w not in words[first]]
    second = None
    for i in range(len(spans)):
        if i == first:
            continue
        s = score(i, left)
        if s[0] >= ALSO_FLOOR and better(s, second):
            second = (i, s)
    return spans[first], (spans[second[0]] if second else None), hits


@dataclass
class SourceMatch:
    same: bool
    expected: str | None
    got: str | None
    #: The sentence alone, citation aside: what an untrained model can be
    #: compared on, since it never learned to cite.
    sentence: bool = False

    def to_json(self) -> dict:
        return {"same": self.same, "sentence": self.sentence, "expected": self.expected, "got": self.got}


def same_source(ordinal: int, text: str, expected: str, got: str) -> SourceMatch:
    """Pigeon's `material::same_source`: the same cited section and the same
    supporting sentence (either's second clause counts); both declining is
    the same, one declining is not."""
    def words(s):
        return text[s[0][0]:s[0][1]] if s else None

    de, dg = declines(expected), declines(got)
    if de and dg:
        return SourceMatch(True, None, None, True)
    e, g = support_scored(text, expected), support_scored(text, got)
    if de or dg:
        return SourceMatch(False, None if de else words(e), None if dg else words(g))
    cited_right = all(n == ordinal for n in cited(got)) and (ordinal not in cited(expected) or ordinal in cited(got))
    sentence_right = bool(e and g and (e[0] == g[0] or e[1] == g[0] or g[1] == e[0]))
    return SourceMatch(cited_right and sentence_right, words(e), words(g), sentence_right)
