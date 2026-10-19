"""Patch Motion: a move as a recipe, drawn as an animated SVG.

No model, no server — it runs in the standing pytest gate.

    python3 -m pytest src-tauri/python/bench/test_patchmotion.py -q
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import patchcorpus as pc  # noqa: E402
import patchmotion as pm  # noqa: E402

NUMBER = re.compile(r"-?\d*\.\d+|-?\d+")


def test_every_move_of_the_sheet_is_a_move_except_the_mark():
    tags = json.loads((pm.SOURCE / "patch-sprites.json").read_text())["meta"]["frameTags"]
    assert set(pm.moves()) == {t["name"] for t in tags} - {"mark"}
    for t in tags:
        if t["name"] != "mark":
            assert pm.moves()[t["name"]]["frames"] == t["to"] - t["from"] + 1


def test_every_move_renders_as_well_formed_svg_with_one_group_per_frame():
    """An ill-formed drawing would not error in Pigeon: the <img> just goes blank."""
    for name, move in pm.moves().items():
        root = ET.fromstring(pm.render_svg({"move": name}))
        frames = [g for g in root.iter("{http://www.w3.org/2000/svg}g") if "f" in g.get("class", "").split()]
        assert len(frames) == move["frames"], name


def test_the_frames_tile_the_cycle_exactly_once():
    """Each frame's visible slice starts where the one before ends, and the last
    runs to 100% — so at every moment exactly one frame shows."""
    for name, move in pm.moves().items():
        css = pm.fragments()["move"][name]
        edges = []
        for i in range(move["frames"]):
            steps = re.search(rf"@keyframes {name}-{i}\{{(.*?)\}}\.", css).group(1)
            shown = [float(p) for p in re.findall(r"([\d.]+)%\{visibility:visible\}", steps)]
            hidden = [float(p) for p in re.findall(r"([\d.]+)%\{visibility:hidden\}", steps) if float(p) > 0]
            assert len(shown) == 1 and len(hidden) <= 1, (name, i, steps)
            edges.append((shown[0], hidden[0] if hidden else 100.0))
        assert edges[0][0] == 0 and edges[-1][1] == 100.0, name
        for (_, end), (start, _) in zip(edges, edges[1:]):
            assert end == start, (name, edges)
        assert f"{pm._seconds(pm.cycle_seconds(name))} step-end infinite" in css


def test_a_move_that_plays_once_holds_its_last_frame_before_it_plays_again():
    hop = pm.moves()["hop"]
    assert not hop["loop"]
    assert pm.cycle_seconds("hop") == hop["frames"] / hop["fps"] + pm.HOLD
    assert pm.cycle_seconds("wave") == pm.moves()["wave"]["frames"] / pm.moves()["wave"]["fps"]


def test_trimming_coordinates_never_joins_or_splits_a_number():
    """Path data runs numbers together (`3.2.5` is 3.2 then .5). Every geometry
    attribute of the sheet must hold the same numbers after trimming, to 0.006."""
    svg = (pm.SOURCE / "patch-sprites.svg").read_text()
    for m in re.finditer(r' (?:d|points|transform)="([^"]*)"', svg):
        before = [float(x) for x in NUMBER.findall(m.group(1))]
        after = [float(x) for x in NUMBER.findall(pm._trim_numbers(m.group(1)))]
        assert len(before) == len(after) and all(abs(a - b) <= 0.006 for a, b in zip(before, after)), m.group(1)[:80]
    assert pm._trim_numbers("L-10.39999.00002-.1 3.2.999 12.0001.5") == "L-10.4 0-.1 3.2 1 12.0.5"


def test_reduced_motion_shows_the_first_frame_still():
    head = pm.head()
    assert "@media (prefers-reduced-motion:reduce){.f{animation:none!important}.f0{visibility:visible}}" in head
    for name in pm.moves():
        assert 'class="f f0 ' in pm.fragments()["move"][name]


def test_the_payload_renders_exactly_what_python_renders():
    payload = pm.skill_payload()
    assert payload["render"] == {"method": "concat", "version": 1}
    for name in pm.moves():
        recipe = {"move": name}
        parts = payload["head"] + "".join(payload["parts"][k][recipe[k]] for k in payload["order"] if recipe.get(k))
        assert payload["wrapper"].replace("{parts}", parts) == pm.render_svg(recipe)
    json.dumps(payload)


def test_a_recipe_is_never_repaired_and_a_wrong_move_fails_the_ride():
    assert pm.loads_recipe('{"move":"waves"}') == (None, ["move='waves' is not a move"])
    assert pm.loads_recipe('{"move":"wave","speed":2}')[0] is None
    assert pm.loads_recipe("wave") == (None, ["not JSON"])
    assert pm.grade_move('{"move":"wave"}', '{"move":"wave"}')[0]
    passed, detail = pm.grade_move('{"move":"wave"}', '{"move":"hop"}')
    assert not passed and "hop where wave" in detail


def test_the_practice_set_is_honest():
    corpus = pc.build()
    assert pc.problems(corpus) == []
    assert pc.build() == corpus, "the set rebuilds byte for byte"
    groups = {r["group"] for r in corpus["holdout"]}
    assert groups == {"seen", "unseen"}


def test_a_motion_pigeon_opens_and_draws_from_its_payload_alone(tmp_path):
    import zipfile
    from bench.test_pigeon import base_q4, run_folder
    from pigeon import pack_run, validate_pigeon
    from pigeon.validate import installed_bases

    run = run_folder(tmp_path, base_q4(tmp_path), labels=False, flight=False)
    state = json.loads((run / "state.json").read_text())
    state["skill"] = {"name": pm.SKILL_NAME, "version": pm.SKILL_VERSION,
                      "fields": list(pm.RECIPE_KEYS), "payload": pm.skill_payload()}
    (run / "state.json").write_text(json.dumps(state))
    out = tmp_path / "patch-motion.pigeon"
    manifest = pack_run(run, out)
    assert manifest["skill"] == {"name": "patch-motion", "version": 1}
    assert validate_pigeon(out, bases=installed_bases([tmp_path])).ok

    with zipfile.ZipFile(out) as z:
        payload = json.loads(z.read("skill/skill.json"))
    parts = payload["head"] + payload["parts"]["move"]["loaf_sleep"]
    assert payload["wrapper"].replace("{parts}", parts) == pm.render_svg({"move": "loaf_sleep"})
