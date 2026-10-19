"""The pigeon look: an icon as a recipe of named parts.

No model, no server — it runs in the standing pytest gate.

    python3 -m pytest src-tauri/python/bench/test_pigeonlook.py -q
"""

from __future__ import annotations

import itertools
import json
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pigeonlook as pl  # noqa: E402

LOOK = {"sky": "harbour", "plumage": "slate", "frame": "walnut", "beak": "amber",
        "marks": "speckles", "eyes": "goggles", "hat": None}


def pixels(png: bytes) -> tuple[int, bytes]:
    """(size, raw RGB) out of the PNG render_png writes: one IDAT, filter 0."""
    size = struct.unpack(">I", png[16:20])[0]
    at, idat = 8, b""
    while at < len(png):
        n = struct.unpack(">I", png[at:at + 4])[0]
        if png[at + 4:at + 8] == b"IDAT":
            idat += png[at + 8:at + 8 + n]
        at += 12 + n
    raw = zlib.decompress(idat)
    return size, b"".join(raw[r * (size * 3 + 1) + 1:(r + 1) * (size * 3 + 1)] for r in range(size))


def at(png: bytes, x: int, y: int) -> str:
    size, rgb = pixels(png)
    i = (y * size + x) * 3
    return "#" + rgb[i:i + 3].hex()


def test_the_same_name_always_draws_the_same_pigeon():
    assert pl.random_look("Coin Sorter") == pl.random_look("Coin Sorter")
    assert len({pl.dumps_look(pl.random_look(f"pigeon {i}")) for i in range(40)}) > 30


def test_every_random_look_is_a_valid_recipe():
    for i in range(200):
        text = pl.dumps_look(pl.random_look(i))
        look, problems = pl.loads_look(text)
        assert problems == [] and look is not None, text


def test_every_part_renders_as_well_formed_svg():
    """Each shape part at least once, against every colour — an ill-formed
    fragment would break the preview silently, since an <img> just goes blank."""
    tables = {**pl.COLOURS, **pl.SHAPES}
    for key in pl.RECIPE_KEYS:
        for name in tables[key]:
            look = {**LOOK, key: name}
            ET.fromstring(pl.render_svg(look))


def test_the_payload_renders_exactly_what_python_renders():
    """The UI preview concatenates the payload; the PNG is drawn from the
    tables. Both must be the same drawing."""
    payload = pl.skill_payload()
    for look in [LOOK, pl.random_look("a"), pl.random_look("b"), pl.random_look("c")]:
        parts = payload["head"] + "".join(payload["parts"][k][look[k]]
                                           for k in payload["order"] if look.get(k))
        assert payload["wrapper"].replace("{parts}", parts) == pl.render_svg(look)
    json.dumps(payload)                     # data only, all the way down


def test_a_look_is_never_repaired():
    assert pl.loads_look('{"sky":"harbour"}')[0] is None
    look, problems = pl.loads_look(pl.dumps_look({**LOOK, "hat": "sombrero"}))
    assert look is None and "hat='sombrero' is not a part" in problems
    look, problems = pl.loads_look(json.dumps({**LOOK, "wings": "big"}))
    assert look is None and "'wings' is not a slot" in problems
    assert pl.loads_look("not json") == (None, ["not JSON"])
    assert pl.loads_look(pl.dumps_look({**LOOK, "marks": None}))[0] is not None


def test_the_png_is_512_and_draws_the_colours_the_look_names():
    """Pixels read back from the PNG, at points that only one part covers."""
    png = pl.render_png(LOOK)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (512, 512)
    assert at(png, 10, 10) == pl.SKY["harbour"]["sky"]
    assert at(png, 300, 480) == pl.PLUMAGE["slate"]["plumage"]
    assert at(png, 170, 310) == pl.FIXED["lens"]
    assert at(png, 430, 310) == pl.BEAK["amber"]["beak"]
    assert at(png, 129, 500) == pl.PLUMAGE["slate"]["mark"]

    other = pl.render_png({**LOOK, "sky": "plum", "beak": "rose", "marks": None})
    assert at(other, 10, 10) == pl.SKY["plum"]["sky"]
    assert at(other, 430, 310) == pl.BEAK["rose"]["beak"]
    assert at(other, 129, 500) == pl.PLUMAGE["slate"]["plumage"], "no marks, bare plumage"


def test_there_is_only_ever_one_eye():
    """A side view: no eyewear may draw a second lens across the face."""
    for name, shapes in pl.EYES.items():
        assert sum(s[-1] == "lens" for s in shapes) <= 1, f"{name} draws a second lens"
        assert sum(s[-1] in ("ink", "pupil") for s in shapes) <= 1, f"{name} draws a second pupil or dark lens"


def test_the_nose_piece_stops_well_short_of_the_beak_tip():
    beak_tip = max(s[1] + s[3] for s in pl.HEAD_SHAPES if s[-1] == "beak")
    for name, shapes in pl.EYES.items():
        front = max((s[1] + s[3]) for s in shapes if s[0] == "rect" and s[-1] == "frame") \
            if any(s[0] == "rect" and s[-1] == "frame" for s in shapes) else 0
        assert front <= beak_tip - 60, f"{name}'s nose piece reaches x {front}, beak tip {beak_tip}"


def test_the_packed_icon_is_the_look_the_step_showed(tmp_path):
    """pack_run with a look writes that look as icon.png; with neither a PNG
    nor a look, the name picks one — the same one the draft offered."""
    import zipfile
    from bench.test_pigeon import base_q4, run_folder
    from pigeon import pack_run

    run = run_folder(tmp_path, base_q4(tmp_path))
    chosen = {**LOOK, "sky": "ember"}
    pack_run(run, tmp_path / "a.pigeon", look=chosen)
    with zipfile.ZipFile(tmp_path / "a.pigeon") as z:
        assert z.read("icon.png") == pl.render_png(chosen)

    manifest = pack_run(run, tmp_path / "b.pigeon")
    with zipfile.ZipFile(tmp_path / "b.pigeon") as z:
        assert z.read("icon.png") == pl.render_png(pl.random_look(manifest["name"]))


def test_a_look_pigeon_opens_and_draws_from_its_payload_alone(tmp_path):
    """No app lists `pigeon-look`. It opens because its payload says it is drawn
    by concatenation, and a foreign reader draws it with nothing but the file."""
    import zipfile
    from bench.test_pigeon import base_q4, run_folder
    from pigeon import pack_run, validate_pigeon
    from pigeon.validate import installed_bases

    run = run_folder(tmp_path, base_q4(tmp_path), labels=False, flight=False)
    state = json.loads((run / "state.json").read_text())
    state["skill"] = {"name": pl.SKILL_NAME, "version": pl.SKILL_VERSION,
                      "fields": list(pl.RECIPE_KEYS), "payload": pl.skill_payload()}
    (run / "state.json").write_text(json.dumps(state))
    out = tmp_path / "icon-maker.pigeon"
    manifest = pack_run(run, out)
    assert manifest["skill"] == {"name": "pigeon-look", "version": 1}
    assert validate_pigeon(out, bases=installed_bases([tmp_path])).ok

    with zipfile.ZipFile(out) as z:
        payload = json.loads(z.read("skill/skill.json"))
    look = pl.random_look("Chess Coach")
    parts = payload["head"] + "".join(payload["parts"][k][look[k]] for k in payload["order"] if look.get(k))
    assert payload["wrapper"].replace("{parts}", parts) == pl.render_svg(look)


def test_every_combination_of_colours_draws():
    for sky, plumage, frame, beak in itertools.product(pl.SKY, pl.PLUMAGE, pl.FRAME, pl.BEAK):
        look = {**LOOK, "sky": sky, "plumage": plumage, "frame": frame, "beak": beak}
        assert pl.loads_look(pl.dumps_look(look))[0] == look
    pl.render_png({**LOOK, "hat": "mortarboard", "eyes": "shades", "marks": "scarf"})
