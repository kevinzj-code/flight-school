"""The emoji face grammar: the parts, the assembler, and the recipe form.

Lifted out of `scripts/build_emoji_corpus.py` so that the corpus builder, the
check ride's grader and the measurement scripts all read ONE definition. A
second copy of these tables would be a second answer to "what does `grin` mean",
and the whole point of a grammar is that there is one.

It lives under `src-tauri/python/` rather than `scripts/` for a concrete reason:
this directory is on the import path for `quickbuild.py`, for `instructor/*` and
for every test under `bench/`, so a grader that runs INSIDE a check ride can
import it and the standing pytest gate covers it. Nothing in `scripts/` gets
either.

Deliberately dependency-free — standard library only — so importing it can never
drag the app's engine, config or model code into a grader.

THE RECIPE. A face can be written two ways: as ~380 characters of SVG, or as the
~70-character object that produced it —

    {"eyes": "happy", "mouth": "grin", "brows": null, "extras": ["halo"]}

The second is a closed vocabulary of about forty names. Measured (the baseline notes):
raw-SVG output is chaotic across runtimes because every coordinate is a
knife-edge token, while the style itself transfers perfectly. Naming the parts
and letting code draw them is the arm of that experiment; `render()` is what
makes a recipe back into a drawing.
"""

from __future__ import annotations

import json
from typing import Any, Callable

HEAD = '<circle cx="12" cy="12" r="9"/>'

# --- eyes -------------------------------------------------------------------
E = {
    "dots":     '<path d="M9 9.5v1.5"/><path d="M15 9.5v1.5"/>',
    "wide":     '<circle cx="9" cy="10" r="1.5"/><circle cx="15" cy="10" r="1.5"/>',
    "happy":    '<path d="M7.5 10.5Q9 8 10.5 10.5"/><path d="M13.5 10.5Q15 8 16.5 10.5"/>',
    "closed":   '<path d="M7.5 9.5Q9 11.5 10.5 9.5"/><path d="M13.5 9.5Q15 11.5 16.5 9.5"/>',
    "squint":   '<path d="M7.5 10h3"/><path d="M13.5 10h3"/>',
    # Kept for "hopeful face", which reads correctly with it; "thinking face"
    # moved off it, because raised dots alone cannot carry scepticism.
    "up":       '<path d="M9 9v1.5"/><path d="M15 9v1.5"/>',
    "wink":     '<path d="M7.5 9.5Q9 11.5 10.5 9.5"/><path d="M15 9.5v1.5"/>',
    "x":        '<path d="M7.5 8.5l3 3"/><path d="M10.5 8.5l-3 3"/>'
                '<path d="M13.5 8.5l3 3"/><path d="M16.5 8.5l-3 3"/>',
    "hearts":   '<path d="M9 11.5Q7 9.5 8 8.5T9 9.5Q10 8 11 8.5T9 11.5"/>'
                '<path d="M15 11.5Q13 9.5 14 8.5T15 9.5Q16 8 17 8.5T15 11.5"/>',
    "spiral":   '<path d="M10.5 10Q10.5 8.5 9 8.5T7.5 10T9 11.5"/>'
                '<path d="M16.5 10Q16.5 8.5 15 8.5T13.5 10T15 11.5"/>',
    "wobble":   '<path d="M7.5 10Q9 8.5 10.5 10"/><path d="M13.5 10Q15 11.5 16.5 10"/>',
    "shades":   '<path d="M6.5 9h11"/><path d="M7.5 9v2.5h3V9"/><path d="M13.5 9v2.5h3V9"/>',
    "glasses":  '<circle cx="9" cy="10" r="2"/><circle cx="15" cy="10" r="2"/><path d="M11 10h2"/>',
}

# --- brows ------------------------------------------------------------------
B = {
    "angry":   '<path d="M7 7l3 1.5"/><path d="M17 7l-3 1.5"/>',
    "worried": '<path d="M7 8l3-1.5"/><path d="M17 8l-3-1.5"/>',
    "raised":  '<path d="M7.5 7.5Q9 6 10.5 7.5"/><path d="M13.5 7.5Q15 6 16.5 7.5"/>',
    # One brow up, one level: the whole reading of a sceptical face is the
    # ASYMMETRY, which a matched pair cannot carry.
    "cocked":  '<path d="M7.5 7.3Q9 6.7 10.5 7.3"/><path d="M13.5 7Q15 5 16.5 7"/>',
}

# --- mouths -----------------------------------------------------------------
M = {
    "smile":       '<path d="M8 14Q12 17.5 16 14"/>',
    "slight":      '<path d="M9.5 14.5Q12 16 14.5 14.5"/>',
    "grin":        '<path d="M7.5 13.5h9Q16 18 12 18T7.5 13.5"/>',
    "open":        '<circle cx="12" cy="15.5" r="2"/>',
    "gasp":        '<ellipse cx="12" cy="15.5" rx="2" ry="2.5"/>',
    "frown":       '<path d="M8 16.5Q12 13 16 16.5"/>',
    "slightfrown": '<path d="M9.5 16Q12 14.5 14.5 16"/>',
    "flat":        '<path d="M9 15.5h6"/>',
    "smirk":       '<path d="M9 15.5Q12 17 15 14.5"/>',
    "wavy":        '<path d="M8.5 15.5q1.5-1.5 3 0t3 0"/>',
    "tongue":      '<path d="M8 14Q12 17.5 16 14"/><path d="M10.5 16v1.5a1.5 1.5 0 0 0 3 0V16"/>',
    # Third attempt, and the one that reads. A D-shape with bars inside fills
    # solid at stroke-width 2 (cut on review, twice). The zip mouth — which
    # survived review — is a line with crossbars, and the reason it works is the
    # WHITE SPACE it leaves. Two rails with a gap between them is the same idiom
    # at mouth scale: upper and lower teeth, with air in the middle.
    "teeth":       '<path d="M7.5 13.5h9"/><path d="M7.5 16.5h9"/>'
                   '<path d="M10.5 13.5v3"/><path d="M13.5 13.5v3"/>',
    "zip":         '<path d="M8 15.5h8"/><path d="M10 14.5v2"/><path d="M14 14.5v2"/>',
    "kiss":        '<ellipse cx="12" cy="15.5" rx="2" ry="1.5"/>',
    "pout":        '<path d="M10.5 16.5q1.5-2 3 0"/>',
    "cry":         '<path d="M9 17Q12 13.5 15 17"/>',
}

# --- extras -----------------------------------------------------------------
X = {
    "tears":  '<path d="M8 12.5L6.5 16"/><path d="M16 12.5L17.5 16"/>',
    "tear1":  '<path d="M16 12.5L17.5 16"/>',
    "sweat":  '<path d="M20 1.5c1.2 1.7 1.9 2.7 1.9 3.4a1.9 1.9 0 0 1-3.8 0c0-.7.7-1.7 1.9-3.4z"/>',
    "blush":  '<path d="M6 13.5h1.5"/><path d="M16.5 13.5H18"/>',
    "halo":   '<ellipse cx="12" cy="1.5" rx="4.5" ry="1.5"/>',
    "horns":  '<path d="M4.5 6.5l1-3 2.5 1.5"/><path d="M19.5 6.5l-1-3-2.5 1.5"/>',
    "zzz":    '<path d="M16.5 3h3l-3 3h3"/>',
    "hat":    '<path d="M8 4.5L12 1l4 3.5"/>',
    "bandage":'<path d="M13.5 5.5l3 2.5"/><path d="M16.5 5l-3 3"/>',
}


def face(eyes: str, mouth: str, brows: str | None = None, extras: tuple = ()) -> str:
    parts = [HEAD, E[eyes]]
    if brows:
        parts.append(B[brows])
    parts.append(M[mouth])
    parts += [X[x] for x in extras]
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
            'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round">' + "".join(parts) + "</svg>")


# Expression families, read off the mouth. Used ONLY to stratify the holdout:
# a random split put all eight held-out designs in the sad family, which tests
# whether the adapter generalises within one mood rather than across the style.
# The packer stratifies its examples for the same reason.
FAMILY = {
    "smile": "positive", "slight": "positive", "grin": "positive",
    "tongue": "positive", "kiss": "positive",
    "frown": "negative", "slightfrown": "negative", "cry": "negative",
    "pout": "negative",
    "open": "surprise", "gasp": "surprise",
    "flat": "flat", "zip": "flat", "teeth": "flat",
    "smirk": "wry", "wavy": "wry",
}


#: The recipe's keys, in the order they are always written. Fixed so that a
#: model writing them in another order can never be the difference between a
#: pass and a fail — `dumps_recipe` is the only serializer and the grader
#: compares parsed objects, not text.
RECIPE_KEYS = ("eyes", "mouth", "brows", "extras")

#: Which table each key draws from. Built from the tables themselves, so a part
#: added to `M` is legal everywhere at once — in the corpus, in the schema the
#: check ride constrains with, and in the grader.
PARTS: dict[str, dict[str, str]] = {"eyes": E, "mouth": M, "brows": B, "extras": X}


def dumps_recipe(recipe: dict[str, Any]) -> str:
    """The canonical text of a recipe: fixed key order, compact separators,
    `brows` always present as null when absent, extras in table order."""
    extras = [x for x in X if x in set(recipe.get("extras") or ())]
    return json.dumps({"eyes": recipe.get("eyes"), "mouth": recipe.get("mouth"),
                       "brows": recipe.get("brows") or None, "extras": extras},
                      separators=(",", ":"))


def loads_recipe(text: str, *, parse: Callable[[str], Any] | None = None
                 ) -> tuple[dict[str, Any] | None, list[str]]:
    """`(recipe, problems)`. Empty problems means it is renderable.

    NEVER repairs. An unknown part name is a miss, not something to coerce to
    the nearest known one — a grader that repairs its input cannot measure the
    thing it was built to measure.

    `parse` exists so a caller that is reading MODEL output can inject
    `instructor.planspec.extract_json`, which is this repo's hardened extractor
    (strips fences, scans braces without being fooled by braces inside strings).
    Injected rather than imported so this module keeps no dependency on the app.
    """
    obj: Any
    try:
        obj = (parse or json.loads)(text.strip())
    except Exception:
        return None, ["not JSON"]
    if not isinstance(obj, dict):
        return None, ["not a JSON object"]

    problems: list[str] = []
    recipe: dict[str, Any] = {}
    for key in ("eyes", "mouth"):
        value = obj.get(key)
        if value not in PARTS[key]:
            problems.append(f"{key}={value!r} is not a part")
        recipe[key] = value
    brows = obj.get("brows") or None
    if brows is not None and brows not in B:
        problems.append(f"brows={brows!r} is not a part")
    recipe["brows"] = brows
    extras = obj.get("extras") or []
    if not isinstance(extras, list):
        problems.append("extras is not a list")
        extras = []
    for x in extras:
        if x not in X:
            problems.append(f"extras={x!r} is not a part")
    recipe["extras"] = [x for x in extras if x in X]
    return (None if problems else recipe), problems


def render(recipe: dict[str, Any]) -> str:
    """A recipe back into SVG, through the same `face()` the corpus was drawn
    with. A recipe that assembles to an off-style SVG is a bug in the grammar,
    and grading a recipe run through the house-style checker is where it shows."""
    return face(recipe["eyes"], recipe["mouth"], recipe.get("brows"),
                tuple(recipe.get("extras") or ()))


def recipe_schema() -> dict[str, Any]:
    """JSON Schema for a recipe, with the enums BUILT FROM the tables.

    Handed to llama-server as `response_format`, which compiles it to a grammar,
    so a constrained check ride cannot produce an unparseable recipe at all. That
    moves the failure from "malformed" to "wrong part" — which is the failure the
    grader is supposed to see."""
    return {
        "type": "object",
        "properties": {
            "eyes": {"type": "string", "enum": sorted(E)},
            "mouth": {"type": "string", "enum": sorted(M)},
            "brows": {"type": ["string", "null"], "enum": [*sorted(B), None]},
            "extras": {"type": "array", "items": {"type": "string", "enum": sorted(X)}},
        },
        "required": list(RECIPE_KEYS),
        "additionalProperties": False,
    }


def diff_recipe(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, bool]:
    """Per-part correctness. `extras` compares as a SET: the order of two
    decorations on a face is not a semantic difference, and scoring it as one
    would report a correct drawing as wrong."""
    return {
        "eyes": expected.get("eyes") == actual.get("eyes"),
        "mouth": expected.get("mouth") == actual.get("mouth"),
        "brows": (expected.get("brows") or None) == (actual.get("brows") or None),
        "extras": set(expected.get("extras") or ()) == set(actual.get("extras") or ()),
    }


def grade_recipe(expected: str, actual: str, *,
                 parse: Callable[[str], Any] | None = None) -> tuple[bool, str]:
    """The check ride's rule for this job: `(passed, detail)`.

    Passes only when the answer parses, names only known parts, and matches the
    expected recipe on every key. `detail` names the parts that differed, so a
    miss is readable in check-ride.json without re-running anything."""
    want, want_problems = loads_recipe(expected, parse=parse)
    if want is None:
        return False, "the expected answer is not a recipe: " + "; ".join(want_problems)
    got, problems = loads_recipe(actual, parse=parse)
    if got is None:
        return False, "; ".join(problems)
    per_part = diff_recipe(want, got)
    wrong = [k for k, ok in per_part.items() if not ok]
    if not wrong:
        return True, "every part matches"
    return False, "wrong: " + ", ".join(f"{k} {want.get(k)!r}≠{got.get(k)!r}" for k in wrong)


#: A part seen fewer than this many times across a corpus is reported by
#: `coverage_problems`. NOT a proven threshold: measured 2026-09-12, 21 of the
#: ~42 parts fall below 3 in the 60-design corpus — and that corpus is the one
#: whose adapter collapsed to 3-9 distinct drawings over 20 prompts. Whether
#: raising coverage FIXES the collapse is the open question this number exists
#: to test, not something it asserts.
MIN_SIGHTINGS = 3


def part_counts(recipes) -> dict[str, dict[str, int]]:
    """How many times each part is used, per key. `recipes` is any iterable of
    recipe dicts — the corpus's own `parts` blocks do."""
    counts: dict[str, dict[str, int]] = {k: {name: 0 for name in PARTS[k]} for k in RECIPE_KEYS}
    for r in recipes:
        for key in ("eyes", "mouth"):
            if r.get(key) in counts[key]:
                counts[key][r[key]] += 1
        if r.get("brows") in counts["brows"]:
            counts["brows"][r["brows"]] += 1
        for x in r.get("extras") or ():
            if x in counts["extras"]:
                counts["extras"][x] += 1
    return counts


def coverage_problems(recipes, *, floor: int = MIN_SIGHTINGS) -> list[str]:
    """Parts a corpus shows too rarely to be learnable, worst first.

    `bench.corpus.label_ratio_problems` cannot do this job and never will: it
    auto-skips whenever the answers are mostly distinct, which is exactly what a
    corpus of unique drawings looks like. So an emoji corpus passes the standing
    audit while half its vocabulary appears once. This is the missing check."""
    counts = part_counts(recipes)
    thin = [(n, key, name) for key, table in counts.items()
            for name, n in table.items() if n < floor]
    thin.sort()
    return [f"{key}={name} appears {n}x, under the {floor} needed to learn it"
            for n, key, name in thin]


#: The name and version an app must recognise to render this Pigeon's answers.
#: Versioned because the part tables are the contract: a Pigeon trained against
#: one set of shapes must not be rendered by another.
SKILL_NAME = "face-grammar"
SKILL_VERSION = 1


def skill_payload() -> dict[str, Any]:
    """Everything needed to turn a recipe into a drawing, as DATA.

    Deliberately not code. A `.pigeon` arrives by AirDrop or flash drive, and an
    app that executed code out of one would have thrown away the entire
    validate-then-load doctrine the format is built on. This grammar does not
    need code: a part name maps to an SVG fragment, and rendering is
    concatenation — the head, then one fragment per named part, inside the
    wrapper. Ten lines in any language, no evaluator.
    """
    return {
        "name": SKILL_NAME,
        "version": SKILL_VERSION,
        # How it is drawn — what a reader accepts a skill by (PIGEON-FORMAT §9.3).
        "render": {"method": "concat", "version": 1},
        "how": "Concatenate `head` and the fragment for each named part, in the "
               "order eyes, brows, mouth, extras, inside `wrapper`.",
        "wrapper": ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
                    'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                    'stroke-width="2" stroke-linecap="round" '
                    'stroke-linejoin="round">{parts}</svg>'),
        "head": HEAD,
        "order": ["eyes", "brows", "mouth", "extras"],
        "parts": {"eyes": E, "brows": B, "mouth": M, "extras": X},
        "recipe_schema": recipe_schema(),
    }
