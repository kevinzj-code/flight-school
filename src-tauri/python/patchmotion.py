"""Patch, moving: an animation as a recipe of one named part.

The pigeon look (`pigeonlook.py`) carried over to motion. A recipe names a move —

    {"move": "wave"}

— and the skill draws it as an animated SVG. Pigeon shows a skill's drawing as
`<img src="data:image/svg+xml,…">`, and an SVG shown that way still animates
by its own CSS (measured 2026-09-16 in WKWebView under Pigeon's exact CSP: CSS
keyframes, SMIL, transforms and an embedded GIF all moved). So a GIF or an MP4
is not needed and would not fit `concat` v1 anyway: the drawing is text.

THE FRAMES ARE PATCH V2'S OWN. `datasets/patch-motion/source/patch-sprites.svg`
is the vector sprite sheet `patch/v2/build.js` writes (one `<g id="move_NN">`
per 320×288 cell, feet on the same ground line), and `patch-sprites.json` its
frame tags (fps, loop). Nothing here redraws Patch; it only times the frames.

HOW A MOVE PLAYS. Every frame is a group, hidden by default. Each frame has its
own keyframes that make it visible for its slice of the cycle, stepped, so
exactly one frame shows at a time. A move that does not loop in the sheet (a
hop, a blink) holds its last frame for HOLD seconds and plays again, the way
the sprite-sheet preview does. Someone who asked for less motion sees the first
frame, still.

Standard library only, like `pigeonlook.py` and `pigeon/pack.py`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[2] / "datasets" / "patch-motion" / "source"
CELL_W, CELL_H = 320, 288
HOLD = 0.8
#: Tags in the sheet that are not motion. The navy mark is a logo, one frame.
NOT_MOVES = ("mark",)

RECIPE_KEYS = ("move",)
SKILL_NAME = "patch-motion"
SKILL_VERSION = 1


def _pct(v: float) -> str:
    return f"{round(v * 100, 3):g}%"


def _seconds(v: float) -> str:
    return f"{round(v, 4):g}s"


_NUM = re.compile(r"-?\d*\.\d+")


def _trim_numbers(markup: str) -> str:
    """Coordinates to 2 decimals — a hundredth of a pixel in a 320-wide cell.
    A number whose next character is `.` keeps its decimal point: in path data
    `12.0.5` is two numbers, and `12.5` would be one."""
    def sub(m: re.Match) -> str:
        glued = markup[m.end():m.end() + 1] == "."
        v = round(float(m.group()), 2)
        if v == int(v):
            # `12` would swallow a following `.5`, so a glued one stays `12.0`.
            text = f"{int(v)}.0" if glued else f"{int(v)}"
        else:
            text = f"{v:.2f}".rstrip("0")
            if text.lstrip("-").startswith("0."):
                text = text.replace("0.", ".", 1)
        # `3.2.5` is two numbers because the second starts with `.`. If this
        # one rounded to something that does not (`.004` -> `0`), a digit
        # would join the one before it, so it gets a separator instead.
        if m.start() and markup[m.start() - 1].isdigit() and text[0] not in ".-":
            text = " " + text
        return text
    return _NUM.sub(sub, markup)


@lru_cache(maxsize=1)
def _sheet() -> tuple[str, dict[str, dict[str, Any]]]:
    """`(style, moves)` from the vendored sheet. `moves[name]` holds `fps`,
    `loop` and `frames`, each frame the markup inside its cell group."""
    svg = (SOURCE / "patch-sprites.svg").read_text(encoding="utf-8")
    tags = json.loads((SOURCE / "patch-sprites.json").read_text(encoding="utf-8"))["meta"]["frameTags"]
    style = re.search(r"<style>(.*?)</style>", svg, re.S).group(1)
    style = re.sub(r"\s+", " ", style)
    style = re.sub(r"\s*([{};,:])\s*", r"\1", style).strip()

    starts = [m for m in re.finditer(r'<g id="([a-z_]+)_(\d+)" transform="translate\(([-\d.]+) ([-\d.]+)\)">', svg)]
    cells: dict[str, list[tuple[int, str, float, float]]] = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else svg.rindex("</svg>")
        chunk = svg[m.end():end]
        assert chunk.endswith("</g>"), f"frame {m.group(1)}_{m.group(2)} is not one group"
        col = int(m.group(2))
        cells.setdefault(m.group(1), []).append(
            (col, chunk[:-4], float(m.group(3)) - col * CELL_W, float(m.group(4)) % CELL_H))
    moves: dict[str, dict[str, Any]] = {}
    for tag in tags:
        if tag["name"] in NOT_MOVES:
            continue
        frames = sorted(cells[tag["name"]])
        assert [c for c, *_ in frames] == list(range(tag["to"] - tag["from"] + 1)), tag["name"]
        moves[tag["name"]] = {"fps": tag["fps"], "loop": tag["loop"], "note": tag.get("note", ""),
                              "offset": (frames[0][2], frames[0][3]),
                              "frames": [_trim_numbers(body) for _, body, _, _ in frames]}
    return style, moves


def moves() -> dict[str, dict[str, Any]]:
    """Every move with its fps, loop flag and the sheet's note, frames left out."""
    return {k: {"fps": v["fps"], "loop": v["loop"], "frames": len(v["frames"]), "note": v["note"]}
            for k, v in _sheet()[1].items()}


def cycle_seconds(name: str) -> float:
    move = _sheet()[1][name]
    n = len(move["frames"])
    return n / move["fps"] + (0 if move["loop"] else HOLD)


def _fragment(name: str) -> str:
    """One move: its frames' timing, then its frames."""
    move = _sheet()[1][name]
    n, fps = len(move["frames"]), move["fps"]
    total = cycle_seconds(name)
    ox, oy = move["offset"]
    css, groups = [], []
    for i, body in enumerate(move["frames"]):
        a, b = i / fps / total, ((i + 1) / fps / total if i < n - 1 else 1.0)
        key = f"{name}-{i}"
        steps = ([] if i == 0 else ["0%{visibility:hidden}"]) + [f"{_pct(a)}{{visibility:visible}}"]
        if b < 1.0:
            steps.append(f"{_pct(b)}{{visibility:hidden}}")
        css.append(f"@keyframes {key}{{{''.join(steps)}}}"
                   f".{key}{{animation:{key} {_seconds(total)} step-end infinite}}")
        groups.append(f'<g class="f f{i} {key}" transform="translate({ox:g} {oy:g})">{body}</g>')
    return f"<style>{''.join(css)}</style>" + "".join(groups)


WRAPPER = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{CELL_W}" height="{CELL_H}" '
           f'viewBox="0 0 {CELL_W} {CELL_H}">{{parts}}</svg>')


def head() -> str:
    """Patch's colours, and the two rules every move relies on: frames start
    hidden, and with reduced motion the first frame stands still."""
    return ("<style>" + _sheet()[0] + ".f{visibility:hidden}"
            "@media (prefers-reduced-motion:reduce){.f{animation:none!important}.f0{visibility:visible}}"
            "</style>")


@lru_cache(maxsize=1)
def fragments() -> dict[str, dict[str, str]]:
    return {"move": {name: _fragment(name) for name in _sheet()[1]}}


def render_svg(recipe: dict[str, Any]) -> str:
    """Exactly what `skill_payload` tells an app to do."""
    parts = fragments()
    return WRAPPER.replace("{parts}", head() + "".join(
        parts[k][recipe[k]] for k in RECIPE_KEYS if recipe.get(k)))


# ---------------------------------------------------------------------------
# the recipe


def dumps_recipe(move: str) -> str:
    return json.dumps({"move": move}, separators=(",", ":"))


def loads_recipe(text: str, *, parse=None) -> tuple[dict[str, Any] | None, list[str]]:
    """`(recipe, problems)`. Never repairs: a move the sheet does not have is a
    miss, not something to coerce to the nearest one."""
    try:
        obj = (parse or json.loads)(text.strip())
    except Exception:
        return None, ["not JSON"]
    if not isinstance(obj, dict):
        return None, ["not a JSON object"]
    problems = [f"{k!r} is not a slot" for k in obj if k not in RECIPE_KEYS]
    move = obj.get("move")
    if not move:
        problems.append("move is missing")
    elif move not in _sheet()[1]:
        problems.append(f"move={move!r} is not a move")
    return (None if problems else {"move": move}), problems


def grade_move(expected: str, actual: str, *, parse=None) -> tuple[bool, str]:
    """The check-ride rule: `(passed, detail)`. Unlike a look, a move HAS a
    right answer — the one its phrase was written for — so a valid recipe of
    the wrong move fails. `detail` says which way it was wrong."""
    got, problems = loads_recipe(actual, parse=parse)
    if got is None:
        return False, "; ".join(problems)
    want, _ = loads_recipe(expected, parse=parse)
    if want is None:
        return True, "valid; the expected answer is not a recipe to compare with"
    if want["move"] == got["move"]:
        return True, f"valid; {got['move']}"
    return False, f"valid, but {got['move']} where {want['move']} was expected"


def recipe_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"move": {"type": "string", "enum": list(_sheet()[1])}},
            "required": ["move"], "additionalProperties": False}


def skill_payload() -> dict[str, Any]:
    return {
        "name": SKILL_NAME,
        "version": SKILL_VERSION,
        "render": {"method": "concat", "version": 1},
        "how": "Concatenate `head` and the fragment for the named move inside `wrapper`. "
               "The result is an animated SVG: show it as an image.",
        "wrapper": WRAPPER,
        "head": head(),
        "order": list(RECIPE_KEYS),
        "parts": fragments(),
        "recipe_schema": recipe_schema(),
    }
