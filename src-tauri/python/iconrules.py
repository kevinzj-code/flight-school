"""The Icon Maker's rules: which words in a Pigeon's description pick which part.

The Icon Maker is a Pigeon whose input is another Pigeon's name and one-line
description, and whose answer is a look (`pigeonlook.py`). What it can learn is
only what its practice set is consistent about, and the emoji Pigeon measured
the failure of having nothing to be consistent about: it learned the style and
0/20 of which parts a name means. So the practice set is labelled by these
rules, reviewed and approved as a table before a row was built (2026-09-13).

Each rule is (slot, part, keywords, synonyms). Keywords appear in training.
Synonyms are split by the corpus builder — some in training, some only in the
held-out test — so the check ride can see whether the model learned the idea
or just the word. Rules are listed in priority: the first rule in a slot whose
word appears wins, so "night coding" is goggles, not shades.

A slot no rule claims — and plumage and frame always — is picked from the
prompt text by `pigeonlook.random_look`, so the same Pigeon always gets the
same look and the practice set rebuilds byte for byte.

Standard library only, like `pigeonlook.py`. One definition, used by the corpus
builder and by the live caller that asks the Icon Maker, so the words a model
was trained on and the words it is asked with cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any

import pigeonlook as pl

RULES: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    ("hat", "mortarboard", ("teach", "tutor", "lesson", "course", "study", "homework"),
                           ("learn", "class", "exam", "quiz", "professor")),
    ("hat", "crown",       ("best", "top", "champion", "winner", "ranking"),
                           ("king", "queen", "premium", "elite")),
    ("hat", "beanie",      ("winter", "cold", "hike", "weather"),
                           ("snow", "ski", "camping", "cosy")),
    ("hat", "bow",         ("gift", "party", "birthday", "cute", "kids"),
                           ("celebrate", "wedding", "invitation", "sweet")),
    ("eyes", "goggles",    ("code", "build", "engineer", "debug", "science", "lab"),
                           ("developer", "program", "experiment", "repair")),
    ("eyes", "monocle",    ("formal", "legal", "finance", "expert", "contract", "tax"),
                           ("lawyer", "accountant", "audit", "investment")),
    ("eyes", "shades",     ("night", "chill", "cool", "summer", "music"),
                           ("dj", "beach", "relax", "vibe")),
    ("eyes", "bare",       ("simple", "plain", "quick", "basic"),
                           ("easy", "tiny", "short", "minimal")),
    ("marks", "scarf",     ("writing", "poem", "story", "author", "book"),
                           ("novel", "poetry", "journal", "essay")),
    ("marks", "wing",      ("travel", "flight", "trip", "delivery"),
                           ("fly", "journey", "commute", "route")),
    ("marks", "speckles",  ("data", "numbers", "stats", "chart"),
                           ("spreadsheet", "metrics", "graph", "count")),
    ("marks", "freckles",  ("garden", "plant", "food", "recipe", "cooking"),
                           ("kitchen", "grow", "bake", "farm")),
    ("sky", "midnight",    ("night", "sleep", "dream", "late"),
                           ("moon", "bedtime", "insomnia")),
    ("sky", "lagoon",      ("ocean", "sea", "water", "swim"),
                           ("boat", "marine", "fish", "surf")),
    ("sky", "pine",        ("nature", "forest", "garden", "outdoor"),
                           ("tree", "eco", "park", "trail")),
    ("sky", "ember",       ("urgent", "alert", "fitness", "energy"),
                           ("workout", "gym", "deadline", "fire")),
    ("beak", "gold",       ("money", "budget", "price", "pay"),
                           ("coin", "invoice", "cost", "savings")),
]


def icon_prompt(name: str, line: str = "") -> str:
    """The Icon Maker's user turn. Whitespace is collapsed so a description
    pasted from a README with line breaks asks the same question."""
    name = " ".join(str(name or "").split())
    line = " ".join(str(line or "").split())
    return f"Pigeon: {name}\nDoes: {line}" if line else f"Pigeon: {name}"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def matches(text: str) -> dict[str, tuple[str, str]]:
    """`{slot: (part, the word that picked it)}` for every slot a rule claims.
    Whole words only: "top" is not in "stop"."""
    words = _words(text)
    out: dict[str, tuple[str, str]] = {}
    for slot, part, keywords, synonyms in RULES:
        if slot in out:
            continue
        hit = next((w for w in (*keywords, *synonyms) if w in words), None)
        if hit:
            out[slot] = (part, hit)
    return out


def look_for(name: str, line: str = "") -> dict[str, Any]:
    """The look the rules give a Pigeon: ruled slots from its words, the rest
    picked from the prompt itself, so the answer is repeatable."""
    look = pl.random_look(icon_prompt(name, line))
    for slot, (part, _) in matches(f"{name} {line}").items():
        look[slot] = part
    return look


def rule_problems() -> list[str]:
    """Every rule names a real slot and part, and no word belongs to two rules
    in the same slot (it could only ever pick the first)."""
    tables = {**pl.COLOURS, **pl.SHAPES}
    problems: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    for slot, part, keywords, synonyms in RULES:
        if slot not in tables or part not in tables[slot]:
            problems.append(f"{slot}={part!r} is not a part")
        for w in (*keywords, *synonyms):
            if (slot, w) in seen and seen[(slot, w)] != part:
                problems.append(f"{w!r} picks both {seen[(slot, w)]} and {part} for {slot}")
            seen[(slot, w)] = part
    return problems
