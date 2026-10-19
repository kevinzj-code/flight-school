"""Using a Pigeon: the parts that need no engine.

    python3 -m pytest src-tauri/python/bench/test_ask.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import facegrammar as fg  # noqa: E402
from pigeon.ask import AskError, render_with_skill  # noqa: E402


def test_a_recipe_renders_from_the_payload_alone():
    """The property the whole skill design rests on: an app needs the file's
    strings and a join, never an evaluator."""
    payload = fg.skill_payload()
    recipe = {"eyes": "happy", "mouth": "grin", "brows": None, "extras": ["halo"]}
    assert render_with_skill(payload, recipe) == fg.render(recipe)


def test_the_declared_order_is_honoured_not_the_recipe_key_order():
    payload = fg.skill_payload()
    a = render_with_skill(payload, {"extras": [], "mouth": "smile", "eyes": "dots",
                                    "brows": "worried"})
    b = render_with_skill(payload, {"eyes": "dots", "brows": "worried",
                                    "mouth": "smile", "extras": []})
    assert a == b == fg.render({"eyes": "dots", "mouth": "smile",
                                "brows": "worried", "extras": []})


def test_a_part_the_payload_does_not_carry_is_refused_never_repaired():
    """§9.3. A renderer that coerced an unknown name to the nearest known one
    could not be trusted to have drawn what the model asked for."""
    with pytest.raises(AskError) as e:
        render_with_skill(fg.skill_payload(),
                          {"eyes": "kaleidoscope", "mouth": "smile",
                           "brows": None, "extras": []})
    assert "kaleidoscope" in str(e.value)


def test_extras_may_be_several_and_are_all_drawn():
    payload = fg.skill_payload()
    out = render_with_skill(payload, {"eyes": "happy", "mouth": "slight",
                                      "brows": None, "extras": ["halo", "blush"]})
    assert payload["parts"]["extras"]["halo"] in out
    assert payload["parts"]["extras"]["blush"] in out


def test_the_shipped_pigeon_carries_a_payload_that_renders(tmp_path):
    """The real file, if this machine has one — otherwise skipped rather than
    silently passing."""
    import json, os, zipfile
    store = Path(os.path.expanduser(
        "~/Library/Application Support/mode.airoplane.flightschool/pigeons"))
    f = store / "emoji-designer.pigeon"
    if not f.exists():
        pytest.skip("no packed emoji Pigeon on this machine")
    with zipfile.ZipFile(f) as z:
        payload = json.loads(z.read("skill/skill.json"))
        manifest = json.loads(z.read("pigeon.json"))
    assert manifest["skill"]["name"] == payload["name"]
    svg = render_with_skill(payload, {"eyes": "dots", "mouth": "smile",
                                      "brows": None, "extras": []})
    assert svg.startswith("<svg") and svg.endswith("</svg>")
