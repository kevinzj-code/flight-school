"""The Icon Maker's practice set, built from `iconrules.RULES`.

Every row is a Pigeon's name and one-line description in, and the look the
rules give it out (`iconrules.look_for`). The split is built to answer one
question on the check ride: did the model learn the idea or the word?

* **seen** — a keyword the training rows use, on a name training never saw;
* **unseen** — each rule's LAST synonym, which no training row contains;
* **none** — a description no rule claims, where any valid look is right.

Training rows never contain a held-out synonym or a held-out name, and the
build refuses to write when a check in `problems` fails — a thin rule is one a
1.5B model cannot learn, and that is cheaper to find here than after a run.

Deterministic: the same seed writes the same files, byte for byte.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import iconrules as ir
import pigeonlook as pl

FIRST = ("Pocket", "Swift", "Clever", "Bright", "Brave", "Lucky", "Handy", "Sunny",
         "Gentle", "Nimble", "Steady", "Merry")
SECOND = ("Pigeon", "Helper", "Buddy", "Scout", "Sidekick", "Pal", "Guide", "Friend",
          "Mate", "Sprout")
# Held-out names: none of these words is in a training name.
FIRST_HELD = ("Quiet", "Plucky", "Curious", "Jolly")
SECOND_HELD = ("Courier", "Companion", "Owl", "Sparrow")

ONE = ("All about {a}.", "A pigeon for {a} questions.", "Helps whenever {a} comes up.",
       "Built around {a}.", "Your {a} sidekick.", "Answers anything about {a}.",
       "Keeps your {a} organised.", "Handy for {a}.")
TWO = ("Helps with {a} and {b}.", "Mixes {a} with {b}.", "A pigeon for {a}, and a little {b}.")
NEUTRAL = ("Rewrites messages so they sound friendlier.", "Turns a messy note into a tidy title.",
           "Summarises long emails into three lines.", "Suggests a polite reply.",
           "Checks spelling and grammar.", "Sorts ideas into a neat list.",
           "Explains jargon without the jargon.", "Keeps a running to-do list.",
           "Picks a name for your project.", "Translates notes into French.",
           "Drafts thank-you notes.", "Tidies up meeting notes.",
           "Reminds you what you promised to do.", "Gives feedback on a pitch.",
           "Turns bullet points into a paragraph.")
NEUTRAL_HELD = ("Writes alt text for images.", "Shortens a sentence without losing it.",
                "Suggests a subject line.", "Turns a rant into a calm message.",
                "Finds the question hiding in an email.", "Renames files sensibly.")

HOLDOUT_SEEN, HOLDOUT_UNSEEN = 10, 14
MIN_RULE_FIRES = 5
MIN_PART_SIGHTINGS = 3


def train_words(rule) -> tuple[str, ...]:
    _, _, keywords, synonyms = rule
    return (*keywords, *synonyms[:-1])


def held_word(rule) -> str:
    return rule[3][-1]


def _row(name: str, line: str, group: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"messages": [
        {"role": "user", "content": ir.icon_prompt(name, line)},
        {"role": "assistant", "content": pl.dumps_look(ir.look_for(name, line))}]}
    if group:
        row["group"] = group
    return row


def build(seed: int = 11) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    lines: list[str] = []
    for rule in ir.RULES:
        for w in rule[2]:
            lines += [t.format(a=w) for t in rng.sample(ONE, 2)]
        for w in rule[3][:-1]:
            lines.append(rng.choice(ONE).format(a=w))
    pool = [w for rule in ir.RULES for w in train_words(rule)]
    for _ in range(40):
        a, b = rng.sample(pool, 2)
        lines.append(rng.choice(TWO).format(a=a, b=b))
    lines += list(NEUTRAL) * 2

    names = [f"{a} {b}" for a in FIRST for b in SECOND]
    rng.shuffle(names)
    rows, seen = [], set()
    for i, line in enumerate(lines):
        # Walk the names until this line is paired with one it has not had.
        for j in range(len(names)):
            name = names[(i + j) % len(names)]
            if (name, line) not in seen:
                seen.add((name, line))
                rows.append(_row(name, line))
                break
    rng.shuffle(rows)
    n_valid = max(1, len(rows) // 10)

    held_names = [f"{a} {b}" for a in FIRST_HELD for b in SECOND_HELD]
    rng.shuffle(held_names)
    holdout = []
    for k, rule in enumerate(rng.sample(ir.RULES, HOLDOUT_SEEN)):
        holdout.append(_row(held_names[k % len(held_names)], rng.choice(ONE).format(a=rng.choice(rule[2])), "seen"))
    for k, rule in enumerate(rng.sample(ir.RULES, HOLDOUT_UNSEEN)):
        holdout.append(_row(held_names[(k + HOLDOUT_SEEN) % len(held_names)], rng.choice(ONE).format(a=held_word(rule)), "unseen"))
    for k, line in enumerate(NEUTRAL_HELD):
        holdout.append(_row(held_names[k], line, "none"))
    return {"train": rows[n_valid:], "valid": rows[:n_valid], "holdout": holdout}


def _prompt(row) -> str:
    return row["messages"][0]["content"]


def problems(corpus: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Everything that would make this set unlearnable or its test dishonest."""
    out = list(ir.rule_problems())
    everything = [r for split in ("train", "valid", "holdout") for r in corpus[split]]
    for r in everything:
        if pl.loads_look(r["messages"][1]["content"])[0] is None:
            out.append(f"not a look: {r['messages'][1]['content']}")
    prompts = [_prompt(r) for r in everything]
    for p, n in Counter(prompts).items():
        if n > 1:
            out.append(f"prompt appears {n} times: {p!r}")

    training = corpus["train"] + corpus["valid"]
    fires = Counter()
    for r in corpus["train"]:
        for slot, (part, _) in ir.matches(_prompt(r)).items():
            fires[(slot, part)] += 1
    for slot, part, _, _ in ir.RULES:
        if fires[(slot, part)] < MIN_RULE_FIRES:
            out.append(f"{slot}={part} fires {fires[(slot, part)]}x in train, under {MIN_RULE_FIRES}")

    sightings = Counter()
    for r in corpus["train"]:
        look = json.loads(r["messages"][1]["content"])
        for key in pl.RECIPE_KEYS:
            if look[key]:
                sightings[(key, look[key])] += 1
    tables = {**pl.COLOURS, **pl.SHAPES}
    for key in pl.RECIPE_KEYS:
        for part in tables[key]:
            if sightings[(key, part)] < MIN_PART_SIGHTINGS:
                out.append(f"{key}={part} appears {sightings[(key, part)]}x in train, under {MIN_PART_SIGHTINGS}")

    held = {held_word(rule) for rule in ir.RULES}
    train_names = {_prompt(r).split("\n")[0] for r in training}
    for r in training:
        leaked = held & ir._words(_prompt(r))
        if leaked:
            out.append(f"a held-out synonym is in training: {sorted(leaked)} in {_prompt(r)!r}")
    for r in corpus["holdout"]:
        if _prompt(r).split("\n")[0] in train_names:
            out.append(f"a held-out row reuses a training name: {_prompt(r)!r}")
    for w in held:
        if any(w in train_words(rule) for rule in ir.RULES):
            out.append(f"{w!r} is held out for one rule and trained for another")
    return out


def write(out: Path, corpus: dict[str, list[dict[str, Any]]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "holdout"):
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in corpus[split]))


def summary(corpus: dict[str, list[dict[str, Any]]]) -> str:
    groups = Counter(r.get("group") for r in corpus["holdout"])
    distinct = len({r["messages"][1]["content"] for r in corpus["train"]})
    return (f"train {len(corpus['train'])}  valid {len(corpus['valid'])}  holdout {len(corpus['holdout'])} "
            f"({', '.join(f'{g} {n}' for g, n in sorted(groups.items()))})\n"
            f"{distinct} distinct looks in train")
