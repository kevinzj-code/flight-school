"""The check ride through the engine (pigeon/engine_ride.py), against a fake
`pigeon-ride` that keeps `--out`'s shape. No model, no GPU.

    python3 -m pytest src-tauri/python/bench/test_engine_ride.py -q
"""
from __future__ import annotations

import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import engine_ride as E  # noqa: E402

FAKE = r'''#!{python}
import json, os, sys, zipfile
args = sys.argv[1:]
out = args[args.index("--out") + 1]
pigeon, holdout = args[-1].split("=")
rows = [json.loads(l) for l in open(holdout) if l.strip()]
m = json.loads(zipfile.ZipFile(pigeon).read("pigeon.json"))
control = m.get("base_only") is True
shift = int(os.environ.get("FAKE_SHIFT", "0")) if control else 0
log = os.environ.get("FAKE_LOG")
if log:
    open(log, "a").write(json.dumps({"pigeon": os.path.basename(pigeon), "rows": rows}) + "\n")
if os.environ.get("FAKE_FAIL"):
    print("pigeon-ride: refused", file=sys.stderr); sys.exit(2)
events = []
for r in rows:
    said = ("base:" if control else "") + "re:" + r["question"] + ("|" + str(len(r["history"])) if r.get("history") else "")
    hist = sum(len(h["pigeon"] or "") for h in r.get("history") or [])
    events.append({"type": "answer", "raw": said, "answer": {"kind": "text", "value": said},
                   "stats": {"prompt_tokens": len(r["question"]) + hist + shift, "total_ms": 3}})
pc = args[args.index("--prefix-cache") + 1]
stamp = "libpigeon 0.1.0 (llama.cpp 26394b4; spec 1; prefix-cache " + ("off" if pc == "0" else pc) + ")"
json.dump({"runtime": os.environ.get("FAKE_RUNTIME", stamp),
           "seconds": 0.5, "load_seconds": 0.1,
           "rides": [{"file": pigeon, "set": "holdout",
                      "rows": [{"question": r["question"], "engine": e["raw"], "event": e} for r, e in zip(rows, events)]}]},
          open(out, "w"))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "pigeon-ride"
    exe.write_text(FAKE.replace("{python}", sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PIGEON_RIDE", str(exe))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "asked.jsonl"))
    return tmp_path


def _pigeon(path: Path, trained: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("pigeon.json", json.dumps({"id": "x", "kind": "persona"}))
        z.writestr("system_prompt.txt", "Be kind.")
        if trained:
            z.writestr("adapter.lora.gguf", b"GGUF")
    return path


def _asked(tmp: Path) -> list[dict]:
    return [json.loads(l) for l in (tmp / "asked.jsonl").read_text().splitlines()]


def test_the_pin_is_the_engine_built():
    p = E.pin()
    assert p["commit"] == "032380d" and p["version"].startswith("libpigeon ")
    assert p["version"].endswith(f"; prefix-cache {p['prefix_cache']})"), "the stamp names the setting ridden at (E26)"


def test_rides_ask_at_the_pinned_prefix_cache_and_refuse_it_off(fake, monkeypatch):
    got = E.ride_with_control(_pigeon(fake / "a.pigeon"), [{"input": "one"}], fake / "w", models=fake)
    assert got["engine"] == E.pin()["version"] and "prefix-cache 2048" in got["engine"]
    monkeypatch.setenv("FAKE_RUNTIME", "libpigeon 0.1.0 (llama.cpp 26394b4; spec 1; prefix-cache off)")
    with pytest.raises(E.EngineRideError, match="not the pinned"):
        E.ride_with_control(_pigeon(fake / "b.pigeon"), [{"input": "one"}], fake / "w2", models=fake)


def test_a_table_row_becomes_a_pigeon_ride_row(tmp_path):
    pic = tmp_path / "a.png"; pic.write_bytes(b"x")
    row = {"input": "q", "expected": {"sentence": 2}, "image": str(pic), "host_values": {"sentence": [0, 1, 2]},
           "sections": [3]}
    got = E.holdout_row(row, history=[{"user": "hi", "pigeon": "hello"}])
    assert got == {"question": "q", "answer": '{"sentence":2}', "image": str(pic.resolve()),
                   "values": {"sentence": [0, 1, 2]}, "history": [{"user": "hi", "pigeon": "hello"}], "sections": [3]}
    assert E.holdout_row({"input": "q"}) == {"question": "q", "answer": None}


def test_the_control_is_the_same_file_without_its_adapter(tmp_path):
    src = _pigeon(tmp_path / "a.pigeon")
    ctl = E.control_file(src, tmp_path / "a.control.pigeon")
    z = zipfile.ZipFile(ctl)
    assert "adapter.lora.gguf" not in z.namelist() and z.read("system_prompt.txt") == b"Be kind."
    assert json.loads(z.read("pigeon.json")) == {"id": "x", "kind": "persona", "base_only": True}


def test_a_ride_and_its_control_through_the_engine(fake):
    src = _pigeon(fake / "a.pigeon")
    rows = [{"input": "one", "expected": "x"}, {"input": "two"}]
    got = E.ride_with_control(src, rows, fake / "work", models=fake)
    assert [E.said(e) for e in got["adapter"]["events"]] == ["re:one", "re:two"]
    assert [E.said(e) for e in got["control"]["events"]] == ["base:re:one", "base:re:two"]
    assert got["engine"] == E.pin()["version"]
    assert [a["pigeon"] for a in _asked(fake)] == ["a.pigeon", "a.control.pigeon"], "the control runs after, whole"


def test_a_control_whose_prompt_differs_is_refused(fake, monkeypatch):
    monkeypatch.setenv("FAKE_SHIFT", "1")
    with pytest.raises(E.EngineRideError, match="prompt differs from the ride's on 2 rows"):
        E.ride_with_control(_pigeon(fake / "a.pigeon"), [{"input": "one"}, {"input": "two"}], fake / "w", models=fake)


def test_a_base_pigeon_has_no_control(fake):
    got = E.ride_with_control(_pigeon(fake / "b.pigeon", trained=False), [{"input": "one"}], fake / "w", models=fake)
    assert got["control"] is None and len(_asked(fake)) == 1


def test_an_engine_that_is_not_the_pinned_one_is_refused(fake, monkeypatch):
    monkeypatch.setenv("FAKE_RUNTIME", "libpigeon 0.0.9 (llama.cpp 000; spec 1)")
    with pytest.raises(E.EngineRideError, match="not the pinned"):
        E.ride_with_control(_pigeon(fake / "a.pigeon"), [{"input": "one"}], fake / "w", models=fake)


def test_a_failed_ride_says_why(fake, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(E.EngineRideError, match="refused"):
        E.ask(_pigeon(fake / "a.pigeon"), [{"input": "one"}], fake / "w", models=fake)


def test_earlier_turns_are_asked_first_with_the_engines_own_replies(fake):
    """§5.6 as_sent: each round sends what the engine actually answered before."""
    rows = [{"input": "and tomorrow?", "turns": ["what tonight", "and after"]}, {"input": "plain"}]
    got = E.ask(_pigeon(fake / "a.pigeon"), rows, fake / "w", models=fake)
    asked = _asked(fake)
    assert [len(a["rows"]) for a in asked] == [1, 1, 2]
    assert asked[1]["rows"][0]["history"] == [{"user": "what tonight", "pigeon": "re:what tonight"}]
    assert asked[2]["rows"][0]["history"] == [{"user": "what tonight", "pigeon": "re:what tonight"},
                                              {"user": "and after", "pigeon": "re:and after|1"}]
    assert "history" not in asked[2]["rows"][1]
    assert [E.said(e) for e in got["events"]] == ["re:and tomorrow?|2", "re:plain"] and got["asks"] == 4


def test_the_engine_flight_grades_with_flight_schools_rules():
    from pigeon import basepack
    m = {"kind": "classifier", "output_contract": {"type": "label", "max_tokens": 6}}
    rows = [{"input": "a", "expected": "YES"}, {"input": "b", "expected": "NO"}, {"input": "c", "expected": "NO"}]
    events = [{"type": "answer", "raw": "YES", "answer": {"kind": "label", "value": "YES"}, "stats": {"total_ms": 7}},
              {"type": "answer", "raw": "maybe", "answer": {"kind": "off_contract", "raw": "maybe"}, "stats": {}},
              {"type": "error", "message": "model_failed"}]
    f = basepack.engine_flight(m, rows, events)
    assert (f["passed"], f["total"], f["off_contract"]) == (1, 3, 2)
    assert f["rows"][0]["ms"] == 7 and f["rows"][2]["error"] == "model_failed"


# --- the Teach flow: fly.run_flight graded on the engine's answers ----------------

def test_teach_rows_are_asked_bare_and_a_readers_section_by_ordinal(tmp_path):
    from instructor import fly
    from pigeon.runner import material_block
    chunk = {"ordinal": 4, "title": "Art. 4", "text": "The term is four years."}
    holdout = [{"question": material_block(chunk) + "\n\nhow long is the term", "answer": "four years"},
               {"question": "unknown passage\n\nwhat", "answer": "x", "image": "a.png"}]
    rows = fly.engine_rows({"grounded": True}, holdout, [chunk])
    assert rows[0] == {"input": "how long is the term", "expected": "four years", "sections": [4]}
    assert rows[1] == {"input": "unknown passage\n\nwhat", "expected": "x", "image": "a.png"}
    assert fly.engine_rows({}, [{"question": "q", "answer": "A"}]) == [{"input": "q", "expected": "A"}]


def test_run_flight_grades_the_engines_answers_and_its_control():
    from instructor import fly
    plan = {"one_line": "x", "output_shape": "a label", "labels": ["YES", "NO"], "grading": "label"}
    holdout = [{"question": "a", "answer": "YES"}, {"question": "b", "answer": "NO"}]
    given = {False: ["YES", "NO"], True: ["NO", "NO"]}
    f = fly.run_flight(None, plan, holdout, max_answer_tokens=6, judge=False, control=True,
                       answers=lambda control: given[control])
    assert (f.passed, f.total, f.control_passed) == (2, 2, 1)
    assert [r.actual for r in f.control_rows] == ["NO", "NO"]


def test_a_control_with_earlier_turns_is_compared_on_its_first_ask(fake):
    """Under as_sent the control's later prompts carry its own replies, so only
    the first ask of each row must match (found on the voices, 2026-10-14)."""
    rows = [{"input": "and then?", "turns": ["first"]}, {"input": "plain"}]
    got = E.ride_with_control(_pigeon(fake / "a.pigeon"), rows, fake / "w", models=fake)
    a, c = got["adapter"], got["control"]
    assert a["events"][0]["stats"]["prompt_tokens"] != c["events"][0]["stats"]["prompt_tokens"]
    assert E.same_prompts(a["first_events"], c["first_events"]) == []
