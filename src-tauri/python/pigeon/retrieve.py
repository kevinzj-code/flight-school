"""Retrieval — spec §9.4: the one section a grounded Pigeon answers from.

A port of the reference definitions in `src-tauri/src/chunker.rs`
(`best_chunk_for_in`, `best_chunk_v2`, `hybrid_scores`), kept to them by the
conformance kit's `retrieve` cases: the Constitution's 176 questions, pinned to
what both apps measured (v1 right on 76, v2 on 117, and Flight School's own
stored hybrid picks). The blend's arithmetic is `embed.hybrid_v1`.

Standard library only. `pick` returns section ordinals: one, none (`[]`, and
then the model is not asked — §9.5), or all of them for `whole`.
"""
from __future__ import annotations

import math

from . import embed

STOP = frozenset("""the and for that with this shall from into upon than them they their there which what who whom
when where how does did have has had been being are was were not nor but any all such may can each one two under over
other out its his her""".split())
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
            "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
            "twentieth"]
V1_FLOOR = 9
V2_K1, V2_B, V2_TITLE_BONUS, V2_PHRASE_BONUS, V2_FLOOR = 1.2, 0.75, 2.0, 2.0, 3.0


def _split(text: str, keep) -> list[str]:
    out, cur = [], []
    for c in text + " ":
        if keep(c):
            cur.append(c)
        elif cur:
            out.append("".join(cur))
            cur = []
    return out


# ── keyword-overlap-v1 ──────────────────────────────────────────────────────

def v1(chunks: list[dict], question: str) -> int | None:
    """The index of the first section with the strictly highest score, or None
    under the floor. Words are 4 bytes or more; a word scores its byte length
    when it occurs anywhere in the lowercased title and text."""
    words = [w for w in _split(question.lower(), str.isalnum) if len(w.encode()) >= 4]
    if not words:
        return None
    best, best_score = None, 0
    for i, c in enumerate(chunks):
        hay = c.get("title", "").lower() + " " + c["text"].lower()
        score = sum(len(w.encode()) for w in words if w in hay)
        if score > best_score:
            best, best_score = i, score
    return best if best_score >= V1_FLOOR else None


# ── keyword-overlap-v2 ──────────────────────────────────────────────────────

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


def v2_tokens(text: str) -> list[str]:
    out = []
    for raw in _split(text.lower(), lambda c: (c.isascii() and c.isalnum()) or c == "-"):
        joined = raw.replace("-", "")
        n = _ordinal_number(raw) or _ordinal_number(joined)
        if n:
            out.append(_roman(n))
        elif len(joined) >= 3 and joined not in STOP:
            out.append(_stem(joined))
    return out


def v2_scores(chunks: list[dict], question: str) -> list[float]:
    """The raw BM25-with-bonuses score of every section, in order, with no floor."""
    docs = []
    for c in chunks:
        words = v2_tokens(f"{c.get('title', '')} {c['text']}")
        tf: dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        docs.append((tf, len(words), set(v2_tokens(c.get("title", ""))), set(zip(words, words[1:]))))
    n = len(docs)
    avg = sum(d[1] for d in docs) / n
    df: dict[str, int] = {}
    for tf, *_ in docs:
        for w in tf:
            df[w] = df.get(w, 0) + 1

    def idf(w: str) -> float:
        f = df.get(w, 0)
        return math.log(1.0 + (n - f + 0.5) / (f + 0.5))

    q = v2_tokens(question)
    words = list(dict.fromkeys(q))
    pairs = list(dict.fromkeys(zip(q, q[1:])))
    scores = []
    for tf, length, title, bigrams in docs:
        s = 0.0
        for w in words:
            f = float(tf.get(w, 0))
            if f > 0.0:
                s += idf(w) * f * (V2_K1 + 1.0) / (f + V2_K1 * (1.0 - V2_B + V2_B * length / avg))
                if w in title:
                    s += V2_TITLE_BONUS * idf(w)
        s += V2_PHRASE_BONUS * sum(1 for p in pairs if p in bigrams)
        scores.append(s)
    return scores


def v2(chunks: list[dict], question: str) -> int | None:
    if not chunks:
        return None
    scores = v2_scores(chunks, question)
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best if scores[best] >= V2_FLOOR else None


# ── every method ────────────────────────────────────────────────────────────

def units_of(embeddings: dict, chunks: list[dict]) -> list[tuple[int, list[float]]]:
    """`material/embeddings.json` as (chunk index, vector) pairs."""
    index = {c["ordinal"]: i for i, c in enumerate(chunks)}
    dims = embeddings["model"]["dims"]
    return [(index[u["ordinal"]], embed.unpack_f16(u["f16"], dims)) for u in embeddings["units"]]


def pick(method: str, chunks: list[dict], question: str, *, units=None, qvec=None) -> list[int]:
    """The ordinals of the sections to show for `question`. `hybrid-v1` needs
    the packed `units` and the question's own vector, already at unit length."""
    if method == "whole":
        return [c["ordinal"] for c in chunks]
    if method == "keyword-overlap-v1":
        i = v1(chunks, question)
    elif method == "keyword-overlap-v2":
        i = v2(chunks, question)
    elif method == "hybrid-v1":
        i = embed.hybrid_v1(chunks, units, qvec, v2_scores(chunks, question))
    else:
        raise ValueError(f"retrieval {method!r} is not one this version can do")
    return [] if i is None else [chunks[i]["ordinal"]]
