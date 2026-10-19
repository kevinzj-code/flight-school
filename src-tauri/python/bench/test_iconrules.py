"""The Icon Maker's rules, its prompt, its checker, and its practice set.

No model, no server — it runs in the standing pytest gate.

    python3 -m pytest src-tauri/python/bench/test_iconrules.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import iconcorpus  # noqa: E402
import iconrules as ir  # noqa: E402
import pigeonlook as pl  # noqa: E402

LOOK = {"sky": "harbour", "plumage": "slate", "frame": "walnut", "beak": "amber",
        "marks": None, "eyes": "goggles", "hat": None}


def test_every_rule_names_a_real_part_and_no_word_is_claimed_twice_in_a_slot():
    assert ir.rule_problems() == []


def test_the_prompt_is_one_fixed_shape():
    assert ir.icon_prompt("Chess Coach", "Explains  openings\nand reviews games.") == \
        "Pigeon: Chess Coach\nDoes: Explains openings and reviews games."
    assert ir.icon_prompt("Chess Coach", "") == "Pigeon: Chess Coach"


def test_whole_words_only_and_the_first_rule_in_a_slot_wins():
    assert "hat" not in ir.matches("stop the timer")          # "top" is not in "stop"
    assert ir.matches("code at night")["eyes"] == ("goggles", "code")
    got = ir.matches("a sleep and ocean helper")
    assert got["sky"] == ("midnight", "sleep")


def test_the_rules_pick_the_ruled_slots_and_the_rest_is_repeatable():
    look = ir.look_for("Pocket Pal", "Helps you study for the exam, on a budget.")
    assert (look["hat"], look["beak"]) == ("mortarboard", "gold")
    assert look == ir.look_for("Pocket Pal", "Helps you study for the exam, on a budget.")
    assert pl.loads_look(pl.dumps_look(look))[0] == look


def test_a_valid_look_passes_the_check_ride_even_when_it_disagrees():
    want = pl.dumps_look(LOOK)
    ok, detail = pl.grade_look(want, pl.dumps_look({**LOOK, "sky": "plum", "hat": "crown"}))
    assert ok and "(5/7)" in detail and "sky" not in detail.split("agrees:")[1]
    assert pl.grade_look(want, want) == (True, "valid; agrees: sky,plumage,frame,beak,marks,eyes,hat (7/7)")
    assert pl.grade_look(want, "a grey pigeon in a crown")[0] is False
    ok, detail = pl.grade_look(want, pl.dumps_look({**LOOK, "hat": "sombrero"}))
    assert not ok and "sombrero" in detail


def test_the_check_ride_resolves_the_rule_the_plan_names():
    from instructor.grade import rule_for
    rule = rule_for({"grader": {"how": "rules", "rule": "pigeonlook:grade_look",
                                "schema": "pigeonlook:recipe_schema"}})
    assert rule.grade(pl.dumps_look(LOOK), pl.dumps_look(LOOK))[0]
    assert set(rule.schema["properties"]) == set(pl.RECIPE_KEYS)


def test_the_practice_set_passes_its_checks_and_rebuilds_byte_for_byte():
    corpus = iconcorpus.build(11)
    assert iconcorpus.problems(corpus) == []
    assert json.dumps(corpus) == json.dumps(iconcorpus.build(11))
    assert 250 <= len(corpus["train"]) + len(corpus["valid"]) <= 340
    groups = [r["group"] for r in corpus["holdout"]]
    assert groups.count("seen") == 10 and groups.count("unseen") == 14 and groups.count("none") == 6


def _lineup(tmp_path, pid, skill=None, added=1):
    import zipfile
    store = tmp_path / "pigeons"
    store.mkdir(exist_ok=True)
    manifest = {"name": pid, "kind": "freeform", **({"skill": {"name": skill, "version": 1}} if skill else {})}
    with zipfile.ZipFile(store / f"{pid}.pigeon", "w") as z:
        z.writestr("pigeon.json", json.dumps(manifest))
    (store / f"{pid}.json").write_text(json.dumps({"id": pid, "added_ms": added}))
    return store


def test_the_icon_maker_is_found_in_the_lineup_by_its_skill(tmp_path):
    from pigeon import store as store_mod
    store = _lineup(tmp_path, "plain", added=5)
    assert store_mod.find_by_skill(store, "pigeon-look") is None
    _lineup(tmp_path, "icon-maker", "pigeon-look", added=10)
    _lineup(tmp_path, "icon-maker-2", "pigeon-look", added=20)
    _lineup(tmp_path, "emoji", "face-grammar", added=30)
    assert store_mod.find_by_skill(store, "pigeon-look")["id"] == "icon-maker-2", "the newest one"


def test_asking_for_a_look_always_answers_with_one(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pigeon import ask as ask_mod
    from pigeon import cli
    args = SimpleNamespace(store=str(tmp_path / "pigeons"), name="Chess Coach", line="Explains openings.",
                           llama_server="llama-server", models=[], timeout=5)
    (tmp_path / "pigeons").mkdir()
    random_look = pl.random_look("Chess Coach")

    out = cli.cmd_look(args)
    assert (out["source"], out["recipe"]) == ("random", random_look) and "no Icon Maker" in out["reason"]

    _lineup(tmp_path, "icon-maker", "pigeon-look")
    asked = {}

    def answers(text):
        def fake(path, prompt, **kw):
            asked.update(prompt=prompt, **kw)
            if isinstance(text, Exception):
                raise text
            return {"answer": text}
        return fake

    monkeypatch.setattr(ask_mod, "ask", answers(ask_mod.AskError("the engine stopped before it was ready")))
    out = cli.cmd_look(args)
    assert out["source"] == "random" and "engine stopped" in out["reason"]
    assert asked["prompt"] == ir.icon_prompt("Chess Coach", "Explains openings.") and asked["timeout_s"] == 5

    monkeypatch.setattr(ask_mod, "ask", answers("A grey pigeon with a monocle."))
    out = cli.cmd_look(args)
    assert out["source"] == "random" and "not a look" in out["reason"]

    drawn = {**LOOK, "eyes": "monocle"}
    monkeypatch.setattr(ask_mod, "ask", answers(pl.dumps_look(drawn)))
    out = cli.cmd_look(args)
    assert (out["source"], out["recipe"], out["pigeon"]) == ("icon-maker", drawn, "icon-maker")


def test_the_checks_catch_a_leaked_synonym_and_a_thin_rule():
    corpus = iconcorpus.build(11)
    held = iconcorpus.held_word(ir.RULES[0])
    corpus["train"].append({"messages": [
        {"role": "user", "content": ir.icon_prompt("Pocket Pal", f"All about {held}.")},
        {"role": "assistant", "content": pl.dumps_look(LOOK)}]})
    assert any("held-out synonym" in p for p in iconcorpus.problems(corpus))

    thin = iconcorpus.build(11)
    thin["train"] = [r for r in thin["train"] if ir.matches(r["messages"][0]["content"]).get("beak") is None]
    assert any("beak=gold fires 0x" in p for p in iconcorpus.problems(thin))
