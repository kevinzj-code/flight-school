"""The pigeon look: a Pigeon's icon as a recipe of named parts.

The same idea as `facegrammar.py`, carried over to icons. A look is a closed
vocabulary of names —

    {"sky": "harbour", "plumage": "slate", "frame": "walnut", "beak": "amber",
     "marks": "speckles", "eyes": "goggles", "hat": null}

— and code draws it. Colours are parts like any other: a colour slot's fragment
is a `<style>` rule, the shape slots only name a class, so every sky goes with
every plumage and every hat without a drawing per combination.

ONE DEFINITION, TWO RENDERINGS. Every shape below is a primitive (rect, circle,
ellipse, polygon). `render_svg` writes them as SVG; `render_png` fills the very
same primitives into a 512×512 PNG. The preview a person picks from and the
icon.png that ships cannot disagree, because neither has its own copy of the
drawing.

The tables are designed so any combination reads: every sky is dark, every
plumage is light, every frame is dark enough to sit on a light head. That is
what lets `random_look` pick blindly.

Standard library only, like `facegrammar.py` and `pigeon/pack.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
import zlib
from typing import Any

SIZE = 512

# ---------------------------------------------------------------------------
# shapes — (kind, *geometry, class)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 24) -> list[tuple[float, float]]:
    """Points along a circle, degrees, SVG orientation (y down)."""
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def rect(x, y, w, h, r, cls):
    return ("rect", x, y, w, h, r, cls)


def circle(cx, cy, r, cls):
    return ("circle", cx, cy, r, cls)


def ellipse(cx, cy, rx, ry, cls):
    return ("ellipse", cx, cy, rx, ry, cls)


def poly(points, cls):
    return ("poly", tuple((round(x, 1), round(y, 1)) for x, y in points), cls)


# Colours no slot chooses: the lens, the glint, and near-black for the dark
# lens of the shades. Pupils come with the frame (see FRAME).
FIXED = {"lens": "#d8e7f0", "shine": "#ffffff", "ink": "#0b1118"}

# The tile, the head, the beak — Patch's close-up, measured off Pigeon's app
# icon (src-tauri/icons/icon.png, tile scaled to 512): head x 91-373 with its
# crown at y 130, beak x 340-468. Side view: ONE eye, always.
HEAD_SHAPES = [
    rect(0, 0, SIZE, SIZE, 0, "sky"),
    poly([(91, SIZE), *_arc(232, 271, 141, 180, 360), (373, SIZE)], "plumage"),
    rect(340, 290, 128, 58, 22, "beak"),
    # Patch's lower beak: a slanted top edge and a rounded tip, not a pill.
    poly([(340, 346), *_arc(420, 377, 22, -90, 90, 12), (340, 399)], "beaklow"),
]

# --- colour slots: name -> {class: colour} ---------------------------------
SKY = {
    "harbour": {"sky": "#16283a"},
    "dusk":    {"sky": "#3b2a4f"},
    "pine":    {"sky": "#1f3d36"},
    "ember":   {"sky": "#5a2a24"},
    "storm":   {"sky": "#2f3a45"},
    "midnight": {"sky": "#111a24"},
    "lagoon":  {"sky": "#0f4c5c"},
    "plum":    {"sky": "#4a1f3d"},
}
PLUMAGE = {
    "slate": {"plumage": "#b4bac0", "mark": "#a58fc0"},
    "dove":  {"plumage": "#e4e6ea", "mark": "#9aa7b4"},
    "sand":  {"plumage": "#d9c7a8", "mark": "#a0775a"},
    "blush": {"plumage": "#e8c4c4", "mark": "#b0707a"},
    "mint":  {"plumage": "#bfdccf", "mark": "#5f9c86"},
    "cloud": {"plumage": "#bcd3e6", "mark": "#6f8fb5"},
    "smoke": {"plumage": "#9aa1a8", "mark": "#d8e7f0"},
}
FRAME = {
    # Each frame carries its pupil: Patch's is the frame brown itself, and a
    # deep shade of the frame keeps that family look without a cherry frame
    # giving the bird a bloodshot eye.
    "walnut":   {"frame": "#42210b", "pupil": "#42210b"},
    "cherry":   {"frame": "#8e2430", "pupil": "#3a0f14"},
    "cobalt":   {"frame": "#2c4f8c", "pupil": "#14254a"},
    "forest":   {"frame": "#2d5a3a", "pupil": "#13291a"},
    "graphite": {"frame": "#2b2f36", "pupil": "#15171b"},
}
BEAK = {
    "amber": {"beak": "#ffb340", "beaklow": "#8f6422"},
    "coral": {"beak": "#ff8a65", "beaklow": "#a24b33"},
    "gold":  {"beak": "#ffd54a", "beaklow": "#a7812a"},
    "rose":  {"beak": "#f49ac1", "beaklow": "#9c4f73"},
}

# --- shape slots: name -> [shapes] ------------------------------------------
MARKS = {
    # Patch's speckles run off the bottom edge; so do these.
    "speckles": [rect(116, 466, 26, 56, 13, "mark"), rect(158, 486, 26, 56, 13, "mark")],
    "freckles": [circle(130, 440, 12, "mark"), circle(166, 464, 12, "mark"), circle(132, 490, 12, "mark")],
    "wing":     [rect(150, 424, 120, 32, 16, "mark"), rect(112, 470, 158, 40, 20, "mark")],
    "scarf":    [rect(84, 408, 296, 54, 22, "mark"), rect(130, 440, 46, 80, 18, "mark")],
}
# The eye sits where Patch's does. The strap's front end stops at x 396 — a
# nose piece as long as the beak was wrong — and every eyewear is side-on.
EYE_X, EYE_Y = 213, 288
EYES = {
    # Patch's own: strap to the back of the head, ring, lens, pupil, glint.
    "goggles": [rect(72, 236, 324, 58, 29, "frame"), circle(EYE_X, EYE_Y, 107, "frame"),
                circle(EYE_X, EYE_Y, 86, "lens"), ellipse(261, EYE_Y, 27, 49, "pupil"),
                circle(177, 253, 19, "shine")],
    # A real monocle: one lens, no strap, a little chain hanging from the rim
    # and trailing back down the chest.
    "monocle": [*[circle(x, y, 7, "frame") for x, y in
                  [(208, 396), (199, 412), (188, 427), (176, 442), (166, 458),
                   (158, 475), (153, 493), (150, 511)]],
                circle(EYE_X, EYE_Y, 98, "frame"),
                circle(EYE_X, EYE_Y, 84, "lens"), ellipse(261, EYE_Y, 27, 49, "pupil"),
                circle(177, 253, 19, "shine")],
    # No eyewear: a solid black eye with a small shine on it.
    "bare":    [circle(EYE_X + 15, EYE_Y - 8, 34, "ink"), circle(EYE_X + 4, EYE_Y - 20, 11, "shine")],
    # Side-on sunglasses: one dark lens, the arm back to the ear, a short bridge.
    # Squashed about 20% from the back of the head, and a touch shorter.
    "shades":  [rect(132, 265, 60, 18, 9, "frame"), rect(300, 265, 92, 18, 9, "frame"),
                rect(165, 238, 159, 100, 44, "ink"),
                rect(186, 256, 48, 14, 7, "shine")],
}
HAT = {
    # The frame colour with a light rim under the board, so it reads on the
    # darkest sky.
    "mortarboard": [rect(168, 104, 128, 48, 10, "frame"),
                    poly([(232, 36), (388, 92), (232, 148), (76, 92)], "lens"),
                    poly([(232, 46), (368, 92), (232, 138), (96, 92)], "frame"),
                    rect(344, 90, 8, 70, 4, "beak"), circle(348, 168, 12, "beak")],
    "crown":       [poly([(152, 152), (152, 68), (192, 108), (232, 52), (272, 108),
                          (312, 68), (312, 152)], "beak"),
                    circle(232, 126, 11, "frame")],
    "beanie":      [poly([*_arc(232, 164, 136, 180, 360)], "mark"),
                    rect(90, 146, 284, 38, 19, "frame"), circle(232, 30, 22, "mark")],
    "bow":         [poly([(300, 138), (238, 94), (238, 182)], "mark"),
                    poly([(300, 138), (362, 94), (362, 182)], "mark"),
                    circle(300, 138, 18, "frame")],
}

#: The recipe's keys, in the order they are written AND drawn. Colour slots
#: first (they only define classes), then marks under the eyes under the hat.
RECIPE_KEYS = ("sky", "plumage", "frame", "beak", "marks", "eyes", "hat")
COLOURS: dict[str, dict[str, dict[str, str]]] = {"sky": SKY, "plumage": PLUMAGE, "frame": FRAME, "beak": BEAK}
SHAPES: dict[str, dict[str, list]] = {"marks": MARKS, "eyes": EYES, "hat": HAT}
#: Slots a look may leave out. Every colour slot is required: a missing one is
#: a class nothing defines, which would draw black.
OPTIONAL = ("marks", "hat")

# ---------------------------------------------------------------------------
# SVG


def _num(v: float) -> str:
    return f"{v:g}"


def _svg(shape) -> str:
    kind, *g, cls = shape
    if kind == "rect":
        x, y, w, h, r = g
        rx = f' rx="{_num(r)}"' if r else ""
        return f'<rect class="{cls}" x="{_num(x)}" y="{_num(y)}" width="{_num(w)}" height="{_num(h)}"{rx}/>'
    if kind == "circle":
        cx, cy, r = g
        return f'<circle class="{cls}" cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(r)}"/>'
    if kind == "ellipse":
        cx, cy, rx, ry = g
        return f'<ellipse class="{cls}" cx="{_num(cx)}" cy="{_num(cy)}" rx="{_num(rx)}" ry="{_num(ry)}"/>'
    (points,) = g
    return f'<polygon class="{cls}" points="{" ".join(f"{_num(x)},{_num(y)}" for x, y in points)}"/>'


def _style(classes: dict[str, str]) -> str:
    return "<style>" + "".join(f".{c}{{fill:{v}}}" for c, v in classes.items()) + "</style>"


WRAPPER = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
           f'viewBox="0 0 {SIZE} {SIZE}">{{parts}}</svg>')
HEAD = _style(FIXED) + "".join(_svg(s) for s in HEAD_SHAPES)


def fragments() -> dict[str, dict[str, str]]:
    """Every part as the SVG text a renderer concatenates."""
    out: dict[str, dict[str, str]] = {}
    for key in RECIPE_KEYS:
        if key in COLOURS:
            out[key] = {name: _style(classes) for name, classes in COLOURS[key].items()}
        else:
            out[key] = {name: "".join(_svg(s) for s in shapes) for name, shapes in SHAPES[key].items()}
    return out


def render_svg(look: dict[str, Any]) -> str:
    """A look as SVG: the head, then one fragment per named part, in order,
    inside the wrapper — exactly what `skill_payload` tells an app to do."""
    parts = fragments()
    return WRAPPER.replace("{parts}", HEAD + "".join(
        parts[k][look[k]] for k in RECIPE_KEYS if look.get(k)))


# ---------------------------------------------------------------------------
# the recipe


def dumps_look(look: dict[str, Any]) -> str:
    return json.dumps({k: look.get(k) or None for k in RECIPE_KEYS}, separators=(",", ":"))


def loads_look(text: str, *, parse=None) -> tuple[dict[str, Any] | None, list[str]]:
    """`(look, problems)`. Never repairs, for the reason `facegrammar.loads_recipe`
    gives: an unknown part is a miss, not something to coerce. `parse` is there
    for a caller reading model output to inject its own extractor, as there."""
    try:
        obj = (parse or json.loads)(text.strip())
    except Exception:
        return None, ["not JSON"]
    if not isinstance(obj, dict):
        return None, ["not a JSON object"]
    tables = {**COLOURS, **SHAPES}
    problems: list[str] = []
    look: dict[str, Any] = {}
    for key in RECIPE_KEYS:
        value = obj.get(key) or None
        if value is None:
            if key not in OPTIONAL:
                problems.append(f"{key} is missing")
        elif value not in tables[key]:
            problems.append(f"{key}={value!r} is not a part")
        look[key] = value
    for key in obj:
        if key not in RECIPE_KEYS:
            problems.append(f"{key!r} is not a slot")
    return (None if problems else look), problems


def grade_look(expected: str, actual: str, *, parse=None) -> tuple[bool, str]:
    """The Icon Maker's check-ride rule: `(passed, detail)`.

    Passes on a VALID look alone — the user's call (2026-09-13): a look has no
    single right answer, and the emoji Pigeon scored 0/20 on exact matches while
    drawing perfectly well. `detail` still names the slots that agree with the
    labelled look, so rule agreement is readable in check-ride.json without
    ever deciding a pass."""
    got, problems = loads_look(actual, parse=parse)
    if got is None:
        return False, "; ".join(problems)
    want, _ = loads_look(expected, parse=parse)
    if want is None:
        return True, "valid; the expected answer is not a look to compare with"
    agree = [k for k in RECIPE_KEYS if want.get(k) == got.get(k)]
    return True, f"valid; agrees: {','.join(agree) or 'none'} ({len(agree)}/{len(RECIPE_KEYS)})"


def random_look(seed: str | int) -> dict[str, Any]:
    """A look picked blindly from the tables. The same seed — a Pigeon's name —
    always gives the same look, so a re-draft does not reshuffle someone's icon."""
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    tables = {**COLOURS, **SHAPES}
    look: dict[str, Any] = {}
    for key in RECIPE_KEYS:
        names: list[Any] = list(tables[key])
        if key == "marks":
            names.append(None)
        elif key == "hat":
            names += [None, None]      # most pigeons go bareheaded
        look[key] = rng.choice(names)
    return look


def recipe_schema() -> dict[str, Any]:
    """JSON Schema with the enums built from the tables, for a trained Pigeon's
    constrained decoding when there is one."""
    tables = {**COLOURS, **SHAPES}
    props: dict[str, Any] = {}
    for key in RECIPE_KEYS:
        if key in OPTIONAL:
            props[key] = {"type": ["string", "null"], "enum": [*tables[key], None]}
        else:
            props[key] = {"type": "string", "enum": list(tables[key])}
    return {"type": "object", "properties": props, "required": list(RECIPE_KEYS),
            "additionalProperties": False}


SKILL_NAME = "pigeon-look"
SKILL_VERSION = 1


def skill_payload() -> dict[str, Any]:
    """Everything needed to draw a look, as data. The UI's preview renders from
    this; it is the same shape as `facegrammar.skill_payload`, so a future
    trained look Pigeon can ship it unchanged."""
    return {
        "name": SKILL_NAME,
        "version": SKILL_VERSION,
        # How it is drawn. Readers accept a skill by this, not by its name, so
        # a Pigeon carrying this payload opens in an app that never heard of it.
        "render": {"method": "concat", "version": 1},
        "how": "Concatenate `head` and the fragment for each named part, in the "
               "order " + ", ".join(RECIPE_KEYS) + ", inside `wrapper`. "
               "marks and hat may be null.",
        "wrapper": WRAPPER,
        "head": HEAD,
        "order": list(RECIPE_KEYS),
        "optional": list(OPTIONAL),
        "parts": fragments(),
        "recipe_schema": recipe_schema(),
    }


# ---------------------------------------------------------------------------
# PNG — the same primitives, filled


def _spans(shape, y: float) -> list[tuple[float, float]]:
    """The x-intervals a shape covers on the horizontal line at `y`."""
    kind, *g, _ = shape
    if kind == "rect":
        x, ry, w, h, r = g
        if not ry <= y < ry + h:
            return []
        dy = max(ry + r - y, y - (ry + h - r), 0) if r else 0
        inset = r - math.sqrt(max(r * r - dy * dy, 0)) if dy else 0
        return [(x + inset, x + w - inset)]
    if kind in ("circle", "ellipse"):
        if kind == "circle":
            cx, cy, rx = g
            ry_ = rx
        else:
            cx, cy, rx, ry_ = g
        t = (y - cy) / ry_
        if abs(t) >= 1:
            return []
        hw = rx * math.sqrt(1 - t * t)
        return [(cx - hw, cx + hw)]
    (points,) = g
    xs = []
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        if (y0 <= y < y1) or (y1 <= y < y0):
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    xs.sort()
    return list(zip(xs[0::2], xs[1::2]))


def _bounds(shape) -> tuple[float, float]:
    kind, *g, _ = shape
    if kind == "rect":
        return g[1], g[1] + g[3]
    if kind == "circle":
        return g[1] - g[2], g[1] + g[2]
    if kind == "ellipse":
        return g[1] - g[3], g[1] + g[3]
    ys = [p[1] for p in g[0]]
    return min(ys), max(ys)


def _hex(c: str) -> bytes:
    return bytes(int(c[i:i + 2], 16) for i in (1, 3, 5))


def render_png(look: dict[str, Any], size: int = SIZE) -> bytes:
    """The look as an RGB PNG, drawn at twice the size and averaged down so the
    curves are smooth. A second of pure Python at 512, once per package."""
    classes = dict(FIXED)
    for key, table in COLOURS.items():
        classes.update(table[look[key]])
    shapes = list(HEAD_SHAPES)
    for key in RECIPE_KEYS:
        if key in SHAPES and look.get(key):
            shapes += SHAPES[key][look[key]]

    k = 2
    big = size * k
    scale = size / SIZE
    buf = bytearray(big * big * 3)
    for shape in shapes:
        colour = _hex(classes[shape[-1]])
        top, bottom = _bounds(shape)
        for row in range(max(0, int(top * scale * k)), min(big, math.ceil(bottom * scale * k) + 1)):
            y = (row + 0.5) / (scale * k)
            for x0, x1 in _spans(shape, y):
                a = max(0, math.ceil(x0 * scale * k - 0.5))
                b = min(big, math.ceil(x1 * scale * k - 0.5))
                if b > a:
                    start = row * big * 3
                    buf[start + a * 3:start + b * 3] = colour * (b - a)

    rows = []
    stride = big * 3
    for row in range(size):
        r1 = buf[row * k * stride:(row * k + 1) * stride]
        r2 = buf[(row * k + 1) * stride:(row * k + 2) * stride]
        out = bytearray(size * 3)
        for c in range(3):
            out[c::3] = bytes((p + q + s + t + 2) >> 2 for p, q, s, t in
                              zip(r1[c::6], r1[c + 3::6], r2[c::6], r2[c + 3::6]))
        rows.append(b"\x00" + bytes(out))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
            + chunk(b"IEND", b""))
