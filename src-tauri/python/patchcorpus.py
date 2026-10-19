"""Patch Motion's practice set: what someone says to Patch, and the move it gets.

The input is whatever a person types to Patch in Pigeon's chat, and the answer
is a recipe (`patchmotion.py`). A Pigeon learns a transformation it is shown
consistently and does not learn vocabulary it is never shown (the baseline notes, the
emoji and Icon Maker runs), so every move has a list of phrases written for it,
and the check ride holds some of them back.

Each move has TRAIN phrases, and HELD phrases that no training row contains.
Held-out rows come in two groups, so the scorecard can tell the two apart:

  * `seen`   — a training phrase in a way of addressing Patch it never saw;
  * `unseen` — a held phrase: the idea in words it was not trained on.

Phrases are the whole instruction; `FRAMES` only vary how Patch is addressed.
The prompt is the person's message as typed — `prompt_template` is `{input}` —
so the words a model was trained on and the words Pigeon hands it are the same.

Standard library only.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import patchmotion as pm

#: move -> (training phrases, held-out phrases). Written so neighbouring moves
#: are told apart by what the person means: `loaf_down` is the act of sitting,
#: `loaf` is staying sat; `sleep` is going to bed, `loaf_sleep` a doze.
PHRASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "idle": (
        ("just hang out", "stand there", "wait for me", "stay right there", "hold on a moment",
         "do nothing", "be still", "just be yourself", "hang around", "stand by",
         "act normal", "stay put", "keep me company", "just chill here"),
        ("idle for a bit", "don't do anything special", "stand around")),
    "blink": (
        ("blink", "blink at me", "give me a blink", "blink your eyes", "do a slow blink",
         "flutter your eyes", "bat your eyelids", "blink once", "are you awake? blink",
         "blink if you understand", "close your eyes for a second", "blink twice"),
        ("give me a wink", "blink blink", "flicker those eyes")),
    "look": (
        ("look around", "look behind you", "check over your shoulder", "what's back there?",
         "keep an eye out", "scan the room", "look for my keys", "search for it",
         "go find it", "where did it go?", "glance back", "keep a lookout"),
        ("peek around", "have a look", "watch your back")),
    "talk": (
        ("say something", "talk to me", "tell me a story", "explain it", "speak up",
         "chat with me", "answer me", "what do you think?", "read it out loud",
         "give me a speech", "tell me a joke", "say that again"),
        ("chatter away", "narrate this", "tell me more")),
    "walk": (
        ("walk", "go for a walk", "take a stroll", "walk over here", "come here",
         "walk with me", "keep walking", "march on", "go on a walk", "step forward",
         "walk around the block", "strut your stuff"),
        ("take a walk", "pace back and forth", "wander over")),
    "peck": (
        ("eat something", "have a snack", "peck at the crumbs", "grab some seeds",
         "you must be hungry", "lunch time", "eat your breakfast", "pick up the crumbs",
         "dinner is served", "have a bite", "peck the ground", "find some food"),
        ("chow down", "nibble on something", "feeding time")),
    "paint": (
        ("paint", "paint a picture", "draw something", "make some art", "design me a logo",
         "sketch it out", "paint the fence", "be creative", "make it pretty",
         "draw me a map", "illustrate this", "doodle for me"),
        ("create a masterpiece", "grab your brush", "colour it in")),
    "hop": (
        ("jump", "hop", "jump for joy", "yay!", "we did it!", "hooray", "let's celebrate",
         "i'm so excited", "hop up and down", "that's amazing news", "happy dance",
         "jump up"),
        ("woohoo", "bounce around", "best day ever")),
    "fly": (
        ("fly", "fly away", "take off", "deliver this message", "fly to the moon",
         "go up in the air", "flap your wings", "send it off", "head out", "leave now",
         "fly home", "carry this letter"),
        ("lift off", "wing it over there", "airmail this")),
    "sleep": (
        ("go to sleep", "goodnight", "time for bed", "go to bed", "lights out", "sleep well",
         "sweet dreams", "it's bedtime", "get some sleep", "night night", "turn in for the night",
         "sleep tight"),
        ("hit the hay", "off to dreamland", "rest your head")),
    "proud": (
        ("be proud", "you nailed it", "great job", "well done", "you did great",
         "show off a little", "you finished it", "that was perfect", "you won",
         "puff your chest", "look how good you did", "i'm proud of you"),
        ("take a bow", "strut with pride", "you crushed it")),
    "loaf_down": (
        ("sit down", "take a seat", "sit", "have a seat", "park yourself", "please sit",
         "sit down right there", "settle down", "get down", "plop down", "sit on the floor",
         "hunker down"),
        ("pull up a chair", "sit yourself down", "drop down")),
    "loaf": (
        ("stay sitting", "keep sitting", "rest there", "relax for a while", "take a break",
         "sit tight", "lounge around", "get comfy", "be a loaf", "just rest",
         "keep resting", "take it easy"),
        ("chill on the couch", "put your feet up", "loaf around")),
    "loaf_sleep": (
        ("take a nap", "doze off", "have a snooze", "you look drowsy", "catch some z's",
         "nap time", "snooze for a bit", "rest your eyes", "power nap", "nod off",
         "a little siesta", "drift off for a bit"),
        ("forty winks", "quick nap", "get sleepy")),
    "glide": (
        ("glide", "soar", "float on the breeze", "coast along", "glide over here",
         "soar through the sky", "ride the wind", "cruise", "drift through the air",
         "float gently", "sail across", "hover there"),
        ("coast on the wind", "soar high", "glide smoothly")),
    "wave": (
        ("hi", "hello", "say hi", "wave", "wave hello", "hey there", "bye", "goodbye",
         "welcome", "say bye", "see you later", "greet everyone"),
        ("howdy", "wave goodbye", "good morning")),
}

#: How a phrase is addressed. Training uses FRAMES, the `seen` held-out rows
#: HELD_FRAMES only, so a seen row is a new message, not a repeat.
FRAMES = ("{p}", "Patch, {p}", "{p}, Patch", "hey Patch, {p}", "{P}!", "ok Patch: {p}",
          "{P} please", "Patch {p}")
HELD_FRAMES = ("Patch — {p}", "{P} :)", "come on Patch, {p}")

ROWS_PER_PHRASE = 3


def message(frame: str, phrase: str) -> str:
    return frame.format(p=phrase, P=phrase[:1].upper() + phrase[1:])


def _row(text: str, move: str, group: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"messages": [{"role": "user", "content": text},
                                        {"role": "assistant", "content": pm.dumps_recipe(move)}]}
    if group:
        row["group"] = group
    return row


def build(seed: int = 11) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = []
    for move, (train, _) in PHRASES.items():
        for phrase in train:
            for frame in rng.sample(FRAMES, ROWS_PER_PHRASE):
                rows.append(_row(message(frame, phrase), move))
    rng.shuffle(rows)
    n_valid = max(1, len(rows) // 10)

    holdout = []
    for move, (train, held) in PHRASES.items():
        holdout.append(_row(message(rng.choice(HELD_FRAMES), rng.choice(train)), move, "seen"))
        for phrase in held:
            holdout.append(_row(message(rng.choice(FRAMES + HELD_FRAMES), phrase), move, "unseen"))
    return {"train": rows[n_valid:], "valid": rows[:n_valid], "holdout": holdout}


def _prompt(row) -> str:
    return row["messages"][0]["content"]


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def problems(corpus: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Everything that would make this set unlearnable or its test dishonest."""
    out: list[str] = []
    if set(PHRASES) != set(pm.moves()):
        out.append(f"phrases cover {sorted(PHRASES)}, the sheet has {sorted(pm.moves())}")
    owner: dict[str, str] = {}
    for move, (train, held) in PHRASES.items():
        for phrase in (*train, *held):
            key = _norm(phrase)
            if key in owner and owner[key] != move:
                out.append(f"{phrase!r} is written for both {owner[key]} and {move}")
            owner[key] = move
    everything = [r for split in ("train", "valid", "holdout") for r in corpus[split]]
    for r in everything:
        if pm.loads_recipe(r["messages"][1]["content"])[0] is None:
            out.append(f"not a recipe: {r['messages'][1]['content']}")
    for p, n in Counter(_prompt(r) for r in everything).items():
        if n > 1:
            out.append(f"prompt appears {n} times: {p!r}")
    counts = Counter(json.loads(r["messages"][1]["content"])["move"] for r in corpus["train"])
    for move in pm.moves():
        if counts[move] < 20:
            out.append(f"{move} has {counts[move]} training rows, under 20")
    training = " | ".join(_norm(_prompt(r)) for r in corpus["train"] + corpus["valid"])
    for move, (_, held) in PHRASES.items():
        for phrase in held:
            if _norm(phrase) in training.split(" | ") or f" {_norm(phrase)} " in f" {training} ":
                out.append(f"held-out phrase {phrase!r} ({move}) is in training")
    return out


def write(out: Path, corpus: dict[str, list[dict[str, Any]]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "holdout"):
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in corpus[split]))


def summary(corpus: dict[str, list[dict[str, Any]]]) -> str:
    groups = Counter(r.get("group") for r in corpus["holdout"])
    return (f"train {len(corpus['train'])}  valid {len(corpus['valid'])}  holdout {len(corpus['holdout'])} "
            f"({', '.join(f'{g} {n}' for g, n in sorted(groups.items()))})  moves {len(PHRASES)}")
