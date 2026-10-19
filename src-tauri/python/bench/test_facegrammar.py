"""The face grammar: the parts, the recipe form, and what a corpus must cover.

These are the rules the emoji experiments rest on. They need no model and no
server, so they run in the standing pytest gate.

    python3 -m pytest src-tauri/python/bench/test_facegrammar.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import facegrammar as fg  # noqa: E402


def test_a_recipe_renders_back_to_the_drawing_it_came_from():
    """The whole recipe arm rests on this: naming the parts loses nothing."""
    recipe = {"eyes": "happy", "mouth": "grin", "brows": None, "extras": ["halo"]}
    assert fg.render(recipe) == fg.face("happy", "grin", None, ("halo",))


def test_every_design_in_the_corpus_survives_the_round_trip():
    parts = [r["parts"] for r in json.loads(
        (Path(__file__).resolve().parents[3] / "datasets/emoji/emoji.json").read_text())]
    for p in parts:
        text = fg.dumps_recipe(p)
        back, problems = fg.loads_recipe(text)
        assert not problems, (p, problems)
        assert fg.render(back) == fg.render(p)


def test_the_schema_enums_are_the_tables_themselves():
    """A part added to a table is legal in the schema at once, or a constrained
    check ride would refuse an answer the corpus contains."""
    s = fg.recipe_schema()["properties"]
    assert set(s["eyes"]["enum"]) == set(fg.E)
    assert set(s["mouth"]["enum"]) == set(fg.M)
    assert set(s["extras"]["items"]["enum"]) == set(fg.X)
    assert None in s["brows"]["enum"] and set(fg.B) <= set(s["brows"]["enum"])


def test_the_serializer_is_canonical_whatever_order_it_is_handed():
    a = fg.dumps_recipe({"extras": ["halo"], "mouth": "grin", "eyes": "happy"})
    b = fg.dumps_recipe({"eyes": "happy", "mouth": "grin", "brows": None, "extras": ["halo"]})
    assert a == b, "key order must never be the difference between a pass and a fail"


def test_an_unknown_part_is_a_miss_and_is_never_repaired():
    bad = json.dumps({"eyes": "sparkly", "mouth": "grin", "brows": None, "extras": []})
    recipe, problems = fg.loads_recipe(bad)
    assert recipe is None
    assert any("sparkly" in p for p in problems)


def test_extras_compare_as_a_set_because_their_order_is_not_meaning():
    want = {"eyes": "dots", "mouth": "smile", "brows": None, "extras": ["halo", "blush"]}
    got = {"eyes": "dots", "mouth": "smile", "brows": None, "extras": ["blush", "halo"]}
    assert all(fg.diff_recipe(want, got).values())


def test_grading_names_the_part_that_differed():
    want = fg.dumps_recipe({"eyes": "dots", "mouth": "grin", "brows": None, "extras": []})
    same = fg.grade_recipe(want, want)
    assert same[0] is True
    wrong = fg.dumps_recipe({"eyes": "dots", "mouth": "smile", "brows": None, "extras": []})
    passed, detail = fg.grade_recipe(want, wrong)
    assert passed is False and "mouth" in detail
    assert fg.grade_recipe(want, "I cannot draw that")[0] is False


def test_coverage_reports_the_parts_a_corpus_shows_too_rarely():
    """The check bench/corpus.py cannot make: label_ratio_problems auto-skips
    when the answers are mostly distinct, which is every drawing corpus."""
    recipes = [{"eyes": "dots", "mouth": "smile", "brows": None, "extras": []}] * 5
    recipes += [{"eyes": "hearts", "mouth": "grin", "brows": None, "extras": ["halo"]}]
    problems = fg.coverage_problems(recipes, floor=3)
    assert any("eyes=hearts appears 1x" in p for p in problems)
    assert not any("eyes=dots" in p for p in problems)
    # Worst first, so a reader fixes the emptiest part rather than the first one.
    assert problems == sorted(problems, key=lambda p: int(p.split("appears ")[1].split("x")[0]))


def test_the_shipped_corpus_is_measurably_thin():
    """A regression guard on the finding that motivated the whole series: half
    the vocabulary appears fewer than three times. When a rebalanced corpus
    lands, this number is what moved."""
    parts = [r["parts"] for r in json.loads(
        (Path(__file__).resolve().parents[3] / "datasets/emoji/emoji.json").read_text())]
    # emoji.json carries one row per NAME; the corpus is 60 designs.
    unique = {fg.dumps_recipe(p) for p in parts}
    assert len(unique) == 60
    problems = fg.coverage_problems([json.loads(u) for u in unique])
    assert len(problems) >= 20, "the 60-design corpus shows 21 parts fewer than 3 times"
