"""THE CROSS-APP CONTRACT for the .pigeon file.

Pigeon copies or imports `pigeon/validate.py`; if they cannot share
the module, they keep their own copy in sync BY THIS FILE. Every case here is
a sentence from spec sheet §3.6–§3.7 turned into an assertion, and the two
corrections in §3.7 are the load-bearing ones:

  - the strict-hash trap: same base, different quantization, WARNS AND LOADS
  - wrong-base refusal: names the model needed, never quietly loads
  - a self-test claims a verdict only for label-shaped kinds

The fixtures build tiny GGUF files with real headers, so the identity rule is
exercised against the same reader the apps use, not against a mock of it.

    python3 -m pytest src-tauri/python/bench/test_pigeon.py -q
"""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from pigeon import (FORMAT_VERSION, base_identity, default_icon_png, pack_run,  # noqa: E402
                    pick_examples, quant_name, read_header, validate_pigeon)
from pigeon.gguf import same_base  # noqa: E402
from pigeon.manifest import build_manifest, label_set  # noqa: E402
from pigeon.validate import (BASE_EXACT, BASE_MISSING, BASE_SAME_OTHER_QUANT,  # noqa: E402
                             BASE_WRONG, installed_bases)


# ---------------------------------------------------------------------------
# fixtures: real GGUF headers on tiny files


def write_gguf(path: Path, kv: dict, kind: str = "model") -> Path:
    """A GGUF with a real header and no tensors. The reader stops after the
    header, so this is enough to be a base or an adapter as far as identity
    goes — which is the whole point: identity lives in the header."""
    kv = {"general.type": kind, **kv}
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(kv))

    def s(x: str) -> bytes:
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    for k, v in kv.items():
        out += s(k)
        if isinstance(v, bool):
            out += struct.pack("<I", 7) + struct.pack("<?", v)
        elif isinstance(v, int):
            out += struct.pack("<I", 4) + struct.pack("<I", v)
        elif isinstance(v, float):
            out += struct.pack("<I", 6) + struct.pack("<f", v)
        else:
            out += struct.pack("<I", 8) + s(str(v))
    path.write_bytes(out + b"\x00" * 64)     # a few bytes of "tensor data"
    return path


QWEN25 = {"general.architecture": "qwen2", "general.name": "Qwen2.5 1.5B Instruct",
          "general.basename": "Qwen2.5", "general.size_label": "1.5B"}
QWEN35 = {"general.architecture": "qwen35", "general.name": "Qwen3.5-4B",
          "general.basename": "Qwen3.5-4B", "general.size_label": "4B"}


def base_q4(d: Path, kv=QWEN25, name="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf") -> Path:
    return write_gguf(d / name, {**kv, "general.file_type": 15})


def base_q8(d: Path, kv=QWEN25, name="Qwen2.5-1.5B-Instruct-Q8_0.gguf") -> Path:
    return write_gguf(d / name, {**kv, "general.file_type": 7, "general.name": "Qwen2.5-1.5B-Instruct"})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


_N = itertools.count()


def run_folder(tmp: Path, base: Path, *, vision=False, labels=True, flight=True) -> Path:
    """A run folder shaped exactly as quickbuild.py leaves one."""
    rd = tmp / f"qb-test-{next(_N)}"; (rd / "export").mkdir(parents=True)
    adapter = write_gguf(rd / "export" / "job.lora.gguf", {"general.architecture": "qwen2", "adapter.type": "lora"}, kind="adapter")
    plan = {"job_name": "coin sorter", "one_line": "Sort coin photos by type.", "modality": "vision" if vision else "text",
            "input_shape": "one photo", "output_shape": "exactly one of: penny, dime, unsure" if labels else "one tidy sentence",
            "max_answer_tokens": 5, "question_groups": [{"name": "clear", "count": 10}], "preset": "quick"}
    state = {"stage": "done", "plan": plan, "adapter": str(adapter), "rows": 30, "creative": False,
             "dataset": {"train": 22, "valid": 2, "holdout": 6},
             "flight": {"passed": 5, "total": 6, "control_passed": 1, "graded_by_model": 0,
                        "paired": {"fixed": 4, "broken": 0, "p": 0.125},
                        "rows": [{"question": "what coin is this?", "expected": "penny", "actual": "penny", "passed": True, "how": "exact"},
                                 {"question": "and this one?", "expected": "dime", "actual": "penny", "passed": False, "how": "exact"}]} if flight else None,
             "audit": {"problems": [], "balance": {"penny": 10, "dime": 9, "unsure": 3}}}
    (rd / "state.json").write_text(json.dumps(state))
    (rd / "base.json").write_text(json.dumps({"student_repo": "x", "student_gguf": str(base), "teacher_gguf": str(base)}))
    (rd / "measured.json").write_text(json.dumps({"answer_tokens_claimed": 5, "answer_tokens_measured": 3, "max_seq_length": 128}))
    rows = [{"question": f"what coin is this? {i}", "answer": ["penny", "dime", "unsure"][i % 3], "group": ["clear", "worn"][i % 2]} for i in range(6)]
    if vision:
        for i, r in enumerate(rows):
            img = rd / f"img{i}.jpg"; img.write_bytes(b"\xff\xd8fake"); r["image"] = str(img)
    (rd / "export" / "holdout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rd


# ---------------------------------------------------------------------------
# the GGUF reader and the identity rule


def test_header_identity_is_read_from_the_file(tmp_path):
    p = base_q4(tmp_path)
    h = read_header(p)
    assert base_identity(h) == {"arch": "qwen2", "name": "Qwen2.5", "size": "1.5B", "finetune": None, "repo": None}
    assert quant_name(h) == "Q4_K_M"


def test_finetune_separates_instruct_from_thinking(tmp_path):
    """Spec §7.2 rule 4 (D12). Qwen3-VL-2B Instruct and Thinking share arch,
    basename and size — checked on the real headers 2026-10-02 — and differ only
    in general.finetune. A Pigeon naming one must not match the other; a
    converter's capitalised "Instruct" is the same finetune; a Pigeon written
    before Draft 3 names none and keeps matching by the triple alone."""
    vl = {"general.architecture": "qwen3vl", "general.basename": "qwen3vl", "general.size_label": "2B"}
    instruct = base_identity(read_header(write_gguf(tmp_path / "i.gguf", {**vl, "general.finetune": "instruct"})))
    thinking = base_identity(read_header(write_gguf(tmp_path / "t.gguf", {**vl, "general.finetune": "thinking"})))
    unsloth = base_identity(read_header(write_gguf(tmp_path / "u.gguf", {**vl, "general.basename": "Qwen3-Vl-2B-Instruct",
                                                                         "general.finetune": "Instruct"})))
    bare = base_identity(read_header(write_gguf(tmp_path / "b.gguf", vl)))
    assert instruct["finetune"] == "instruct"
    assert same_base(instruct, unsloth) and not same_base(instruct, thinking)
    assert not same_base(instruct, bare), "a base whose header names no finetune does not match one that does"
    assert same_base({**instruct, "finetune": None}, thinking), "a pre-Draft-3 Pigeon skips the rule"


def test_same_base_survives_a_different_quantization_and_naming_noise(tmp_path):
    """Q4_K_M and Q8_0 of one model, written by converters that spell
    general.name differently, are ONE base."""
    a = base_identity(read_header(base_q4(tmp_path)))
    b = base_identity(read_header(base_q8(tmp_path)))
    assert same_base(a, b)


def test_a_different_model_is_not_the_same_base(tmp_path):
    a = base_identity(read_header(base_q4(tmp_path)))
    b = base_identity(read_header(base_q4(tmp_path, QWEN35, "Qwen3.5-4B-Q4_K_M.gguf")))
    assert not same_base(a, b)


def test_a_different_size_of_the_same_family_is_not_the_same_base():
    assert not same_base({"arch": "qwen2", "name": "Qwen2.5", "size": "1.5B"},
                         {"arch": "qwen2", "name": "Qwen2.5", "size": "7B"})


def test_not_a_gguf_is_refused_by_name(tmp_path):
    p = tmp_path / "x.gguf"; p.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="not a GGUF"):
        read_header(p)


# ---------------------------------------------------------------------------
# the manifest is written from the run, never by hand


def test_manifest_reads_base_identity_and_hash_from_the_run(tmp_path):
    base = base_q4(tmp_path)
    m, facts = build_manifest(run_folder(tmp_path, base))
    assert m["format_version"] == FORMAT_VERSION
    assert m["id"] == "coin-sorter" and m["kind"] == "classifier"
    assert m["base_model"]["file"] == base.name
    assert m["base_model"]["sha256"] == sha(base)
    assert m["base_model"]["arch"] == "qwen2" and m["base_model"]["quant"] == "Q4_K_M"
    assert m["prompt_template"] == "{input}"
    # measured 3 + 4, floor 8; `constrained` says how the ride decoded (this one: free)
    assert m["output_contract"] == {"type": "label", "max_tokens": 8, "constrained": False}
    assert m["temperature"] == 0
    assert m["example_count"] == 24
    assert m["signature"] == ""                                          # reserved, empty in v1
    assert facts["labels"] == ["penny", "dime", "unsure"]


def test_a_vision_run_names_the_image_in_its_template(tmp_path):
    m, _ = build_manifest(run_folder(tmp_path, base_q4(tmp_path), vision=True))
    assert m["prompt_template"] == "{image}\n{input}"
    assert m["base_model"]["vision"] is True and m["inputs"] == ["image", "text"]


def test_a_free_text_job_is_freeform_with_a_measured_budget(tmp_path):
    m, _ = build_manifest(run_folder(tmp_path, base_q4(tmp_path), labels=False))
    assert m["kind"] == "freeform"
    assert m["output_contract"]["type"] == "text"
    assert m["output_contract"]["max_tokens"] >= 16


def test_a_voice_is_not_called_a_persona_when_no_prompt_was_trained_alongside(tmp_path):
    """persona means 'inject system_prompt.txt on every call'. Quick Build
    strips the system prompt before training, so the adapter works with none;
    calling it a persona would ship a prompt it never saw."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    st = json.loads((rd / "state.json").read_text()); st["creative"] = True
    (rd / "state.json").write_text(json.dumps(st))
    m, _ = build_manifest(rd)
    assert m["kind"] == "freeform"


def test_a_missing_base_refuses_to_guess(tmp_path):
    rd = run_folder(tmp_path, tmp_path / "gone.gguf")
    with pytest.raises(FileNotFoundError, match="not on disk"):
        build_manifest(rd)


def test_label_set_matches_the_check_rides_rule():
    assert label_set({"output_shape": "exactly one of: a, b or c"}) == ["a", "b", "c"]
    assert label_set({"output_shape": "one tidy sentence"}) is None


# ---------------------------------------------------------------------------
# the container


def test_the_ride_copy_carries_no_report_and_the_report_names_the_engine(tmp_path):
    """DEV-6: the copy the engine rides has no report, whatever state.json
    holds from an earlier attempt; a flight that names its engine (D98) puts
    it in report.json beside the runtime."""
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base)
    copy = tmp_path / "ride.pigeon"
    pack_run(rd, copy, ride_copy=True)
    assert "report.json" not in zipfile.ZipFile(copy).namelist()
    st = json.loads((rd / "state.json").read_text())
    st["flight"]["engine"], st["flight"]["runtime"] = "libpigeon 0.1.0 (llama.cpp 26394b4; spec 1)", "llama.cpp"
    (rd / "state.json").write_text(json.dumps(st))
    out = tmp_path / "coin.pigeon"
    pack_run(rd, out)
    ride = json.loads(zipfile.ZipFile(out).read("report.json"))["check_ride"]
    assert ride["engine"].startswith("libpigeon") and ride["runtime"] == "llama.cpp"


def test_pack_writes_the_spec_layout_at_the_root_in_store_mode(tmp_path):
    base = base_q4(tmp_path)
    out = tmp_path / "coin-sorter.pigeon"
    pack_run(run_folder(tmp_path, base), out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert {"pigeon.json", "adapter.lora.gguf", "labels.txt", "icon.png", "README.md", "report.json"} <= set(names)
        assert all("/" not in n or n.startswith("examples/") for n in names), "files at the root, no wrapper folder"
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
        assert z.read("labels.txt").decode() == "penny\ndime\nunsure\n"
        report = json.loads(z.read("report.json"))
        assert report["check_ride"]["control_passed"] == 1                 # the control travels
        assert "practice" not in json.dumps(report).lower() or "rows" in report["practice_set"]
        ex = [n for n in names if n.startswith("examples/")]
        assert 3 * 2 <= len(ex) <= 5 * 2


def test_the_training_data_never_leaves(tmp_path):
    """report.json describes the set; it never carries it — and neither does
    the card: its examples are passed check-ride rows, never practice rows."""
    rd = run_folder(tmp_path, base_q4(tmp_path))
    practice = ["a practice-only question that must never ship"]
    (rd / "export" / "train.jsonl").write_text(json.dumps({"question": practice[0], "answer": "penny"}) + "\n")
    out = tmp_path / "x.pigeon"
    pack_run(rd, out)
    with zipfile.ZipFile(out) as z:
        report = json.loads(z.read("report.json"))
        assert report["practice_set"]["rows"] == 30
        assert "rows.jsonl" not in z.namelist()
        assert "train.jsonl" not in z.namelist()
        assert all(not n.startswith("data/") for n in z.namelist())
        manifest_text = z.read("pigeon.json").decode()
    assert practice[0] not in manifest_text


# ---------------------------------------------------------------------------
# the card (PIGEON-FORMAT §2.1)


def test_a_run_packs_a_card_from_its_plan_and_its_passed_check_ride_rows(tmp_path):
    rd = run_folder(tmp_path, base_q4(tmp_path))
    out = tmp_path / "coins.pigeon"
    manifest = pack_run(rd, out)
    card = manifest["card"]
    assert card["does"] == "Sort coin photos by type."
    assert card["input_hint"] == "one photo" and card["output_noun"] == "label"
    assert card["check"] == {"passed": 5, "total": 6} and card["revision"] == 1
    flight = json.loads((rd / "state.json").read_text())["flight"]
    passed = {r["question"] for r in flight["rows"] if r["passed"]}
    assert card["examples"] and {e["input"] for e in card["examples"]} <= passed, "examples are passed check-ride rows only"
    assert card["input_stats"]["words"]["p50"] >= 1 and card["input_stats"]["lines"]["p95"] == 1
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok, r.errors


def test_a_lesson_card_uses_its_approved_pairs_and_has_no_check(tmp_path):
    ld = lesson_folder(tmp_path, base_q4(tmp_path))
    # The lesson's own one-liner is 61 characters: packed as it is, the file
    # goes without a card rather than with a cut sentence, and says why.
    plain, facts = build_manifest(ld, check_hash=False)
    assert "card" not in plain and facts["card_note"] and "at most 60" in facts["card_note"]
    pack_run(ld, tmp_path / "no-card.pigeon")
    manifest = pack_run(ld, tmp_path / "scout.pigeon", does="Decides what the app should do with a message.")
    card = manifest["card"]
    assert card["does"] == "Decides what the app should do with a message."
    assert "check" not in card and "output_noun" not in card
    assert 3 <= len(card["examples"]) <= 5 and card["examples"][0]["input"].startswith("Message:")


def test_an_over_long_does_is_refused_by_name_and_never_cut(tmp_path):
    rd = run_folder(tmp_path, base_q4(tmp_path))
    long = "Sorts every coin photo it is given into penny, dime or unsure, carefully"
    manifest, facts = build_manifest(rd, check_hash=False, does=long)
    assert "card" not in manifest
    assert any("card.does is" in p and "at most 60" in p for p in facts["card_problems"])
    with pytest.raises(ValueError, match="card cannot be packed"):
        pack_run(rd, tmp_path / "too-long.pigeon", does=long)
    assert not (tmp_path / "too-long.pigeon").exists()


def test_a_manifest_without_a_card_still_validates(tmp_path):
    src = tmp_path / "with.pigeon"
    pack_run(run_folder(tmp_path, base_q4(tmp_path)), src)

    def strip_card(name, data):
        if name == "pigeon.json":
            m = json.loads(data); m.pop("card", None)
            return True, json.dumps(m).encode()
        return True, data
    old = tmp_path / "old.pigeon"
    rezip(src, old, strip_card)
    assert validate_pigeon(old, installed_bases([tmp_path])).ok


def test_a_bad_card_is_skipped_with_a_warning_and_the_pigeon_still_loads(tmp_path):
    """Spec §11 (D8): the card is display text only. One that breaks the rules
    is skipped with a warning naming the field — never a refusal."""
    src = tmp_path / "with.pigeon"
    pack_run(run_folder(tmp_path, base_q4(tmp_path)), src)

    def break_card(name, data):
        if name == "pigeon.json":
            m = json.loads(data); m["card"]["check"] = {"passed": 9, "total": 6}; m["card"]["surprise"] = "kept"
            return True, json.dumps(m).encode()
        return True, data
    bad = tmp_path / "bad.pigeon"
    rezip(src, bad, break_card)
    r = validate_pigeon(bad, installed_bases([tmp_path]))
    assert r.ok, r.errors
    assert "card_skipped" in r.warning_codes and any("card.check" in w for w in r.warnings)
    assert not any("surprise" in w for w in r.warnings), "unknown card keys are skipped, not reported"


def test_a_persona_card_can_say_how_it_talks(tmp_path):
    from pigeon.manifest import build_card
    card, problems = build_card(plan={"one_line": "A patient maths tutor."}, kind="persona", flight=None, source=None,
                                pool=[], talks_like="asks one question at a time")
    assert not problems and card["talks_like"] == "asks one question at a time"
    card, problems = build_card(plan={"one_line": "A sorter."}, kind="classifier", flight=None, source=None,
                                pool=[], talks_like="ignored for a sorter")
    assert "talks_like" not in card


def test_a_sealed_pigeon_with_a_card_still_verifies(tmp_path):
    from pigeon import seal
    from bench import make_seal_fixture as fixture
    out = tmp_path / "sealed.pigeon"
    pack_run(run_folder(tmp_path, base_q4(tmp_path)), out)
    key = tmp_path / "test.key"
    key.write_text(fixture.TEST_SEED.hex() + "\n")
    key.chmod(0o600)
    kid = seal.seal_file(out, key)
    assert seal.check_file(out, {kid: fixture.TEST_PUB.hex()}) == kid
    with zipfile.ZipFile(out) as z:
        assert "card" in json.loads(z.read("pigeon.json"))


def test_examples_are_stratified_and_swap_with_the_seed():
    holdout = [{"question": f"q{i}", "answer": "a", "group": ["x", "y", "z"][i % 3]} for i in range(12)]
    a = pick_examples(holdout, n=3, seed=1)
    assert {e["group"] for e in a} == {"x", "y", "z"}
    b = pick_examples(holdout, n=3, seed=2)
    assert [e["question"] for e in a] != [e["question"] for e in b]
    assert len(pick_examples(holdout, n=9)) == 5 and len(pick_examples(holdout, n=1)) == 3


def test_the_default_icon_is_a_512_png():
    png = default_icon_png("CS")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (512, 512)


def test_a_vision_pack_carries_the_sample_photos_and_the_projector_hash(tmp_path):
    base = base_q4(tmp_path, QWEN35, "Qwen3.5-4B-Q4_K_M.gguf")
    rd = run_folder(tmp_path, base, vision=True)
    mm = tmp_path / "Qwen3.5-4B-mmproj-F16.gguf"; mm.write_bytes(b"proj")
    (rd / "config.toml").write_text(f'[models]\ntarget_mmproj = "{mm}"\n')
    out = tmp_path / "v.pigeon"
    m = pack_run(rd, out)
    assert m["base_model"]["mmproj"] == {"file": mm.name, "sha256": sha(mm)}
    with zipfile.ZipFile(out) as z:
        assert any(n.endswith(".jpg") for n in z.namelist())
        assert any(n.endswith(".prompt.txt") for n in z.namelist())
        # The projector travels in the file, and is the bytes the hash names.
        assert z.read("projector.gguf") == b"proj"
        assert z.getinfo("projector.gguf").compress_type == zipfile.ZIP_STORED
        assert "actions.json" not in z.namelist(), "no actions.json in the run, none in the file"
    # Imported pictures never carry an actions.json (spec §13.2, D71): the pack
    # is refused, and the last good file stays as it was.
    (rd / "actions.json").write_text('{"format_version": 1, "keys": {"Confirm": {"code": 7, "name": "X", "hold_ms": 95}}, '
                                     '"frames": {"stack": 1, "gap_ms": 250, "max_side": 768}}')
    before = out.read_bytes()
    with pytest.raises(ValueError, match="cannot be played"):
        pack_run(rd, out)
    assert out.read_bytes() == before and not out.with_suffix(".pigeon.part").exists()
    # Recorded pictures do.
    (rd / "table.json").write_text(json.dumps({"sessionsIn": ["Session 1"]}))
    pack_run(rd, out)
    with zipfile.ZipFile(out) as z:
        assert json.loads(z.read("actions.json"))["keys"]["Confirm"]["code"] == 7
    # One every reader would skip (§13.2) is never packed: no frames, a newer
    # format, or frames no host can follow. The last good file stays as it was.
    before = out.read_bytes()
    for bad in ('{"format_version": 1, "keys": {}}',
                '{"format_version": 2, "keys": {}, "frames": {"stack": 1, "gap_ms": 250, "max_side": 768}}',
                '{"format_version": 1, "keys": {}, "frames": {"stack": 5, "gap_ms": 250, "max_side": 768}}',
                '{"format_version": 1, "keys": {}, "frames": {"stack": 2, "max_side": 768}}',
                '{"format_version": 1, "keys": {}, "frames": {"stack": 2, "gap_ms": -250, "max_side": 768}}',
                '{"format_version": 1, "keys": {}, "frames": {"stack": 1, "gap_ms": 250, "max_side": 768.5}}'):
        (rd / "actions.json").write_text(bad)
        with pytest.raises(ValueError, match="could never play"):
            pack_run(rd, out)
        assert out.read_bytes() == before and not out.with_suffix(".pigeon.part").exists()


# ---------------------------------------------------------------------------
# the import check — the contract itself


def packed(tmp_path, kv=QWEN25, name="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", **kw) -> tuple[Path, Path]:
    base = base_q4(tmp_path, kv, name)
    out = tmp_path / "t.pigeon"
    pack_run(run_folder(tmp_path, base, **kw), out)
    return out, base


def fixed_prompt_check(run_dir, manifest, facts, example):
    """Stands in for the training venv, which a test machine does not have."""
    return {"input": example["question"].strip(), "canonical": "<|im_start|>user\n" + example["question"]}, None


def test_the_packagers_own_gate_accepts_what_it_wrote(tmp_path):
    base = base_q4(tmp_path)
    out = tmp_path / "t.pigeon"
    pack_run(run_folder(tmp_path, base), out, prompt_check_builder=fixed_prompt_check)
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok, r.errors
    assert r.base_match == BASE_EXACT and not r.warnings, r.warnings


def test_a_pigeon_without_a_buildable_prompt_check_says_why(tmp_path):
    """§8.6: a Writer MUST write prompt_check. When the training venv cannot
    render it, the file still packs, the packer says why, and every reader
    warns `prompt_unchecked` rather than trusting a prompt nothing proved."""
    base = base_q4(tmp_path)
    out, why = tmp_path / "t.pigeon", []
    m = pack_run(run_folder(tmp_path, base), out, warnings=why)
    assert "prompt_check" not in m and any("no prompt_check" in w for w in why)
    assert validate_pigeon(out, installed_bases([tmp_path])).warning_codes == ["prompt_unchecked"]


def test_draft_4_fields_are_written_from_the_run(tmp_path):
    """Spec Draft 4: temperature, trained_from, prompt_check, constrained and the
    report's per-ride record all come from the run, never typed."""
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base)
    st = json.loads((rd / "state.json").read_text())
    st["flight"].update(runtime="llama.cpp", constrained=True, temperature=0)
    (rd / "state.json").write_text(json.dumps(st))
    out = tmp_path / "t.pigeon"
    m = pack_run(rd, out, prompt_check_builder=fixed_prompt_check)
    assert m["temperature"] == 0 and m["trained_from"] == {"repo": "x"}
    assert m["output_contract"]["constrained"] is True and m["prompt_check"]["canonical"].startswith("<|im_start|>")
    assert "finetune" not in m["base_model"], "written only when the base's header has one"
    with zipfile.ZipFile(out) as z:
        rep = json.loads(z.read("report.json"))
    ride = rep["check_ride"]
    assert ride["runtime"] == "llama.cpp" and ride["file_sha256"] == m["base_model"]["sha256"]
    assert ride["constrained"] is True and ride["temperature"] == 0 and ride["passes"] == 1
    assert rep["check_rides"] == [{"runtime": "llama.cpp", "file_sha256": sha(base), "passed": 5, "total": 6,
                                   "control_passed": 1, "temperature": 0, "constrained": True}]


def test_finetune_is_written_when_the_base_header_has_it(tmp_path):
    base = base_q4(tmp_path, {**QWEN25, "general.finetune": "Instruct"})
    m, _ = build_manifest(run_folder(tmp_path, base))
    assert m["base_model"]["finetune"] == "Instruct"


def test_a_reasoning_table_packs_as_reasoned_with_its_labels(tmp_path):
    """D26: Flight School's "Reasoning, then a function" answers
    <think>…</think><do>LABEL</do>. It packs as output type `reasoned` with a
    label answer, its labels from the table's labels.txt, and a budget that
    covers the reasoning — never the 8–16 token classifier clamp."""
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base, labels=False)
    st = json.loads((rd / "state.json").read_text())
    st["plan"]["grader"] = {"how": "rules", "rule": "instructor.rules_table:reasoned"}
    (rd / "state.json").write_text(json.dumps(st))
    (rd / "measured.json").write_text(json.dumps({"answer_tokens_measured": 60}))
    (rd / "labels.txt").write_text("KEY(Up)\nKEY(Down)\n")
    out = tmp_path / "t.pigeon"
    m = pack_run(rd, out, prompt_check_builder=fixed_prompt_check)
    assert m["kind"] == "classifier"
    assert m["output_contract"] == {"type": "reasoned", "answer": "label", "max_tokens": 90, "constrained": False}
    with zipfile.ZipFile(out) as z:
        assert z.read("labels.txt").decode().split() == ["KEY(Up)", "KEY(Down)"]
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok and r.self_test == "verdict", r.errors


def test_a_label_check_ride_is_constrained_to_the_label_set():
    from instructor.fly import label_grammar
    assert label_grammar(["KEY(Up)", "KEY(Down)"]) == 'root ::= "KEY(Up)" | "KEY(Down)"'


def test_the_conformance_kit_cases_pass():
    """Flight School against the spec's conformance kit (§16.2): every case, as
    the kit's own harness compares it. The kit lives beside the spec; point
    PIGEON_CONFORMANCE_KIT at it to run this."""
    import os
    import subprocess
    kit = os.environ.get("PIGEON_CONFORMANCE_KIT")
    if not kit or not Path(kit, "run.py").exists():
        pytest.skip("PIGEON_CONFORMANCE_KIT is not set")
    adapter = Path(__file__).resolve().parent.parent / "pigeon" / "conformance.py"
    import json as _json

    def same(want, got, key=None):
        if isinstance(want, dict):
            absent = want.get("$absent", [])
            return (isinstance(got, dict) and not any(k in got for k in absent)
                    and all(k in got and same(v, got[k], k) for k, v in want.items() if k != "$absent"))
        if isinstance(want, list) and isinstance(got, list):
            if key == "warnings":
                return sorted(want) == sorted(got)
            return len(want) == len(got) and all(same(w, g) for w, g in zip(want, got))
        return want == got

    cases = sorted(Path(kit, "cases").glob("*/*/case.json"))
    assert cases, "the kit has no cases"
    failures = []
    for case in cases:
        out = subprocess.run([sys.executable, str(adapter), str(case.parent)], capture_output=True, text=True).stdout
        try:
            got = _json.loads(out)
        except ValueError:
            got = {"error": out}
        want = _json.loads((case.parent / "expected.json").read_text())
        if not same(want, got):
            failures.append((f"{case.parent.parent.name}/{case.parent.name}", want, got))
    assert not failures, failures


def test_not_a_zip_is_rejected(tmp_path):
    p = tmp_path / "x.pigeon"; p.write_bytes(b"hello")
    r = validate_pigeon(p)
    assert not r.ok and "not a ZIP" in r.errors[0]


def test_a_zip_without_a_manifest_is_rejected(tmp_path):
    p = tmp_path / "x.pigeon"
    with zipfile.ZipFile(p, "w") as z: z.writestr("README.md", "hi")
    r = validate_pigeon(p)
    assert not r.ok and "pigeon.json is missing" in r.errors[0]


def rezip(src: Path, dst: Path, edit):
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zo:
        for n in zi.namelist():
            data = zi.read(n)
            keep, data = edit(n, data)
            if keep: zo.writestr(n, data)


# ---------------------------------------------------------------------------
# a hand-graded lesson, packaged from export/pigeon-source.json


def lesson_folder(tmp: Path, base: Path) -> Path:
    """Shaped as lesson.rs::lesson_package leaves one: the adapter beside an
    export/ folder holding pigeon-source.json — no plan, no measured, no
    scored ride."""
    ld = tmp / "lesson-test"; (ld / "export").mkdir(parents=True)
    adapter = write_gguf(ld / "scout.lora.gguf", {"general.architecture": "qwen2", "adapter.type": "lora"}, kind="adapter")
    pairs = [{"question": f"Message: what is {i} percent of 340?", "answer": f"SKILL: calculator | 340 * 0.{i:02d}", "category": ["skill", "none", "search"][i % 3]}
             for i in range(12)]
    (ld / "export" / "pigeon-source.json").write_text(json.dumps({
        "name": "Pilot v4", "adapter": str(adapter), "base_gguf": str(base),
        "base_repo": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "one_line": "Reads one message and decides what the app should do with it.",
        "input_shape": "one message from the person",
        "example_count": 761, "examples": pairs, "checked_by": "asking",
    }))
    return ld


def test_a_lesson_packages_from_its_source_file_with_no_scorecard(tmp_path):
    base = base_q4(tmp_path)
    ld = lesson_folder(tmp_path, base)
    manifest, facts = build_manifest(ld, trained_with="Flight School 0.1.0")
    assert manifest["kind"] == "freeform" and manifest["prompt_template"] == "{input}"
    assert manifest["base_model"]["arch"] == "qwen2" and manifest["base_model"]["file"] == base.name
    assert manifest["example_count"] == 761
    assert facts["flight"] is None and facts["source"]["checked_by"] == "asking"
    out = tmp_path / "scout.pigeon"
    pack_run(ld, out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "report.json" not in names, "no flight, no scorecard"
        assert sum(n.endswith(".expected.txt") for n in names) == 4
        readme = z.read("README.md").decode()
    assert "checked by asking, not scored" in readme
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok and r.self_test == "samples" and r.base_match == BASE_EXACT


def test_a_newer_format_is_refused_and_names_the_version(tmp_path):
    out, _ = packed(tmp_path); newer = tmp_path / "n.pigeon"
    def edit(n, d):
        if n == "pigeon.json":
            m = json.loads(d); m["format_version"] = FORMAT_VERSION + 1; return True, json.dumps(m)
        return True, d
    rezip(out, newer, edit)
    r = validate_pigeon(newer)
    assert not r.ok
    assert f"format {FORMAT_VERSION + 1}" in r.errors[0] and "Update the app" in r.errors[0]


def test_a_missing_required_field_is_named(tmp_path):
    out, _ = packed(tmp_path); bad = tmp_path / "b.pigeon"
    def edit(n, d):
        if n == "pigeon.json":
            m = json.loads(d); del m["prompt_template"]; return True, json.dumps(m)
        return True, d
    rezip(out, bad, edit)
    r = validate_pigeon(bad)
    assert any("'prompt_template'" in e for e in r.errors)


def test_a_classifier_without_labels_txt_is_rejected(tmp_path):
    out, _ = packed(tmp_path); bad = tmp_path / "b.pigeon"
    rezip(out, bad, lambda n, d: (n != "labels.txt", d))
    r = validate_pigeon(bad)
    assert any("labels.txt is required for kind 'classifier'" in e for e in r.errors)


def test_a_persona_without_its_prompt_is_rejected(tmp_path):
    out, _ = packed(tmp_path, labels=False); bad = tmp_path / "b.pigeon"
    def edit(n, d):
        if n == "pigeon.json":
            m = json.loads(d); m["kind"] = "persona"; return True, json.dumps(m)
        return True, d
    rezip(out, bad, edit)
    r = validate_pigeon(bad)
    assert any("system_prompt.txt is required for kind 'persona'" in e for e in r.errors)


def test_a_template_without_the_input_slot_is_rejected(tmp_path):
    out, _ = packed(tmp_path); bad = tmp_path / "b.pigeon"
    def edit(n, d):
        if n == "pigeon.json":
            m = json.loads(d); m["prompt_template"] = "Question:"; return True, json.dumps(m)
        return True, d
    rezip(out, bad, edit)
    assert any("{input}" in e for e in validate_pigeon(bad).errors)


# --- the two corrections ------------------------------------------------


def test_the_strict_hash_trap_same_base_other_quant_warns_and_loads(tmp_path):
    """Spec §3.7 correction 1, and the verification plan's own test: import a
    .pigeon trained against a Q4_K_M onto a machine holding only the Q8_0 of
    the same base. It warns and loads. If it refuses, the correction was
    never applied and the format is rejecting files that work."""
    out, q4 = packed(tmp_path)
    other = tmp_path / "other-mac"; other.mkdir()
    base_q8(other)                                  # only the Q8_0 lives there
    r = validate_pigeon(out, installed_bases([other]))
    assert r.ok, r.errors
    assert r.base_match == BASE_SAME_OTHER_QUANT
    base_warnings = [w for w, c in zip(r.warnings, r.warning_codes) if c == "same_base_other_quant"]
    assert len(base_warnings) == 1 and "Q8_0" in base_warnings[0] and "loads" in base_warnings[0]


def test_the_exact_file_is_the_fast_path_no_warning(tmp_path):
    out, q4 = packed(tmp_path)
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.base_match == BASE_EXACT and "same_base_other_quant" not in r.warning_codes


def test_a_wrong_base_refuses_and_names_the_model_needed(tmp_path):
    """Never load onto a different base; the message names the one needed."""
    out, _ = packed(tmp_path)
    other = tmp_path / "other-mac"; other.mkdir()
    base_q4(other, QWEN35, "Qwen3.5-4B-Q4_K_M.gguf")  # a genuinely different base
    r = validate_pigeon(out, installed_bases([other]))
    assert not r.ok and r.base_match == BASE_WRONG
    assert "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" in r.errors[0]
    assert "will not load onto a different base" in r.errors[0]
    assert r.base_needed == "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"


def test_no_models_at_all_is_missing_not_wrong(tmp_path):
    """The app offers a download for MISSING; the refusal reads the same."""
    out, _ = packed(tmp_path)
    empty = tmp_path / "empty"; empty.mkdir()
    r = validate_pigeon(out, installed_bases([empty]))
    assert not r.ok and r.base_match == BASE_MISSING


def test_the_hash_is_never_the_refusal_rule(tmp_path):
    """A manifest whose sha256 matches nothing on disk, but whose identity
    matches, still loads. The hash speeds up the yes; it never makes the no."""
    out, base = packed(tmp_path)
    base.write_bytes(base.read_bytes() + b"\x00")   # same header, different bytes
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok and r.base_match == BASE_SAME_OTHER_QUANT


def test_self_test_verdict_only_for_label_kinds(tmp_path):
    """Spec §3.7 correction 2: classifier and router get pass/fail, everything
    else gets samples. A green tick that means nothing is worse than no tick."""
    out, _ = packed(tmp_path)
    assert validate_pigeon(out).self_test == "verdict"
    out2, _ = packed(tmp_path, labels=False)
    r = validate_pigeon(out2)
    assert r.ok and r.self_test == "samples"


def test_validation_without_a_machine_context_still_checks_the_file(tmp_path):
    """`bases=None` means "I only want to know if the file is well formed" —
    the base step is skipped, not failed."""
    out, _ = packed(tmp_path)
    r = validate_pigeon(out)
    assert r.ok and r.base_match is None and r.examples >= 3


# ---------------------------------------------------------------------------
# grounded: a Pigeon that answers from material handed to it at ask time
#
# The two rules these cases exist to hold, both of which fail QUIETLY if a
# reader gets them wrong:
#   - the material slot and the grounded kind imply each other, both ways
#   - a retrieval method a reader cannot perform refuses; it never falls back


def add_material(rd: Path, *, sections: int = 3, chars: int = 400,
                 file: str = "manual.md") -> Path:
    """`material/chunks.json` exactly as chunker.rs serialises a ChunkIndex."""
    d = rd / "material"
    d.mkdir(parents=True, exist_ok=True)
    chunks = [{"ordinal": i + 1, "file": file, "title": f"Section {i + 1}",
               "text": f"body {i} " + "x" * chars, "est_tokens": chars // 3}
              for i in range(sections)]
    (d / "chunks.json").write_text(json.dumps(
        {"chunks": chunks, "total_est_tokens": sections * (chars // 3)}))
    return d


def test_a_grounded_run_puts_the_material_slot_above_the_question(tmp_path):
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd)
    m, facts = build_manifest(rd, check_hash=False)
    assert m["kind"] == "grounded"
    # Position is part of the contract: material, blank line, then the question.
    assert m["prompt_template"] == "{material}\n\n{input}"
    assert m["material"]["sections"] == 3
    assert facts["material_dir"] == str(rd / "material")


def test_a_reader_is_searched_section_by_section_whatever_its_size(tmp_path):
    """Flight School's Reader trains one section per row; pasting the whole
    document in at ask time would be a shape the adapter never saw."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd, sections=2, chars=200)
    state = json.loads((rd / "state.json").read_text())
    state["plan"]["kind"] = "reader"
    (rd / "state.json").write_text(json.dumps(state))
    m, _ = build_manifest(rd, check_hash=False)
    assert m["kind"] == "grounded"
    assert m["material"]["retrieval"] == "keyword-overlap-v2"


def add_embeddings(rd: Path, *, model_sha=None, stale=False) -> Path:
    """`material/embeddings.json` as embed.py writes it: one unit per paragraph,
    made with the one embedding model this version names."""
    from pigeon import embed
    index = json.loads((rd / "material" / "chunks.json").read_text())
    units = [{"ordinal": c["ordinal"], "part": 0, "f16": embed.pack_f16([1.0] + [0.0] * (embed.MODEL["dims"] - 1))}
             for c in index["chunks"]]
    model = dict(embed.MODEL)
    if model_sha:
        model["sha256"] = model_sha
    data = {"model": model, "chunks_sha256": "0" * 64 if stale else embed.chunks_digest(index), "units": units}
    (rd / "material" / "embeddings.json").write_text(json.dumps(data))
    return rd / "material" / "embeddings.json"


def test_a_reader_with_embedded_material_packs_as_hybrid_v1_and_carries_the_vectors(tmp_path):
    from pigeon import embed
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd, sections=3)
    add_embeddings(rd)
    m, _ = build_manifest(rd, check_hash=False)
    assert m["material"]["retrieval"] == "hybrid-v1"
    assert m["material"]["embedding"]["sha256"] == embed.MODEL["sha256"]
    assert m["material"]["embedding"]["units"] == 3
    out = tmp_path / "hybrid.pigeon"
    pack_run(rd, out, name="Reader")
    with zipfile.ZipFile(out) as z:
        assert "material/embeddings.json" in z.namelist()
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok, r.errors


def test_stale_or_foreign_embeddings_fall_back_to_the_lexical_search(tmp_path):
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd, sections=3)
    state = json.loads((rd / "state.json").read_text())
    state["plan"]["kind"] = "reader"
    (rd / "state.json").write_text(json.dumps(state))
    add_embeddings(rd, stale=True)
    assert build_manifest(rd, check_hash=False)[0]["material"]["retrieval"] == "keyword-overlap-v2"
    add_embeddings(rd, model_sha="a" * 64)
    assert build_manifest(rd, check_hash=False)[0]["material"]["retrieval"] == "keyword-overlap-v2"
    add_embeddings(rd)
    assert build_manifest(rd, check_hash=False)[0]["material"]["retrieval"] == "hybrid-v1"


def test_hybrid_v1_without_its_vectors_or_with_another_model_is_refused(tmp_path):
    from pigeon import embed
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd, sections=3)
    add_embeddings(rd)
    out = tmp_path / "hybrid.pigeon"
    pack_run(rd, out, name="Reader")
    # The vectors taken out: the method has nothing to search with.
    stripped = tmp_path / "stripped.pigeon"
    with zipfile.ZipFile(out) as z, zipfile.ZipFile(stripped, "w", zipfile.ZIP_STORED) as w:
        for i in z.infolist():
            if i.filename != "material/embeddings.json":
                w.writestr(i, z.read(i.filename))
    r = validate_pigeon(stripped, [])
    assert not r.ok and any("embeddings.json is missing" in e for e in r.errors), r.errors
    # A unit made with another model: refused by name, never searched with this one.
    foreign = tmp_path / "foreign.pigeon"
    with zipfile.ZipFile(out) as z, zipfile.ZipFile(foreign, "w", zipfile.ZIP_STORED) as w:
        for i in z.infolist():
            data = z.read(i.filename)
            if i.filename == "material/embeddings.json":
                e = json.loads(data); e["model"]["sha256"] = "b" * 64; data = json.dumps(e).encode()
            w.writestr(i, data)
    r = validate_pigeon(foreign, [])
    assert not r.ok and any("update the app" in e for e in r.errors), r.errors


def test_material_that_fits_is_pasted_whole_and_material_that_does_not_is_searched(tmp_path):
    small = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(small, sections=2, chars=200)
    assert build_manifest(small, check_hash=False)[0]["material"]["retrieval"] == "whole"

    big = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(big, sections=40, chars=4000)
    assert build_manifest(big, check_hash=False)[0]["material"]["retrieval"] == "keyword-overlap-v2"


def test_the_lesson_refs_folder_is_read_as_material(tmp_path):
    """A lesson that already ran has its chunks under refs/, not material/ —
    a grounded Pigeon can be packed from one without moving anything."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    refs = add_material(rd)
    refs.rename(rd / "refs")
    assert build_manifest(rd, check_hash=False)[0]["kind"] == "grounded"


def test_a_grounded_pack_carries_no_labels_file_because_its_contract_is_text(tmp_path):
    """A run with a label set that also has material packs as grounded, whose
    output_contract is `text`. A labels.txt riding along would claim a contract
    the manifest denies. Found on a real 210-section book, not in a fixture."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=True)
    add_material(rd)
    out = tmp_path / "labelled-grounded.pigeon"
    m = pack_run(rd, out)
    assert m["kind"] == "grounded" and m["output_contract"]["type"] == "text"
    with zipfile.ZipFile(out) as z:
        assert "labels.txt" not in z.namelist()
    assert validate_pigeon(out, bases=installed_bases([tmp_path])).ok


def test_a_grounded_pack_carries_its_material_and_still_not_its_training_rows(tmp_path):
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base, labels=False)
    add_material(rd)
    (rd / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "secret"}]}\n')
    out = tmp_path / "grounded.pigeon"
    pack_run(rd, out)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "material/chunks.json" in names
        assert not [n for n in names if "train" in n or "valid" in n]
        assert json.loads(z.read("material/chunks.json"))["chunks"][0]["ordinal"] == 1
    r = validate_pigeon(out, bases=installed_bases([tmp_path]))
    assert r.ok, r.errors
    # Grounded output is prose: it can be shown, never scored by the importer.
    assert r.self_test == "samples"


def test_a_grounded_pigeon_without_its_material_is_rejected(tmp_path):
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base, labels=False)
    add_material(rd)
    out = tmp_path / "hollow.pigeon"
    pack_run(rd, out)
    stripped = tmp_path / "stripped.pigeon"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(stripped, "w", zipfile.ZIP_STORED) as dst:
        for n in src.namelist():
            if n != "material/chunks.json":
                dst.writestr(n, src.read(n))
    r = validate_pigeon(stripped, bases=installed_bases([tmp_path]))
    assert not r.ok
    assert any("material/chunks.json" in e for e in r.errors)


def _repacked(src: Path, dst: Path, edit) -> Path:
    """Rewrite a .pigeon's manifest through `edit`, keeping every other file."""
    with zipfile.ZipFile(src) as z, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as out:
        for n in z.namelist():
            data = z.read(n)
            if n == "pigeon.json":
                data = json.dumps(edit(json.loads(data))).encode()
            out.writestr(n, data)
    return dst


def test_a_grounded_pigeon_with_no_material_slot_is_rejected(tmp_path):
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd)
    out = tmp_path / "g.pigeon"; pack_run(rd, out)

    def drop_slot(m):
        m["prompt_template"] = "{input}"
        return m
    r = validate_pigeon(_repacked(out, tmp_path / "noslot.pigeon", drop_slot),
                        bases=installed_bases([tmp_path]))
    assert not r.ok
    assert any("{material}" in e and "grounded" in e for e in r.errors)


def test_the_material_slot_on_a_kind_that_carries_none_is_rejected(tmp_path):
    """The quiet failure this prevents: a reader that does not fill the slot
    sends the literal text "{material}" to the model, which answers fluently
    from nothing."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    out = tmp_path / "plain.pigeon"; pack_run(rd, out)

    def add_slot(m):
        m["prompt_template"] = "{material}\n\n{input}"
        return m
    r = validate_pigeon(_repacked(out, tmp_path / "slotted.pigeon", add_slot),
                        bases=installed_bases([tmp_path]))
    assert not r.ok
    assert any("only a grounded Pigeon" in e for e in r.errors)


def test_a_retrieval_method_this_version_cannot_do_is_refused_and_named(tmp_path):
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd)
    out = tmp_path / "g2.pigeon"; pack_run(rd, out)

    def future_method(m):
        m["material"]["retrieval"] = "embedding-cosine-v2"
        return m
    r = validate_pigeon(_repacked(out, tmp_path / "future.pigeon", future_method),
                        bases=installed_bases([tmp_path]))
    assert not r.ok
    assert any("embedding-cosine-v2" in e and "update the app" in e for e in r.errors)


def test_a_run_that_says_where_its_rows_came_from_is_believed_over_the_default(tmp_path):
    """A report.json is a claim about provenance, and a wrong one is worse than
    none. Found by packing a hand-drawn, hand-reviewed corpus and reading back
    "drafted by a local LLM from a brief"."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    state = json.loads((rd / "state.json").read_text())
    state["practice_source"] = "drawn by hand and reviewed one by one"
    (rd / "state.json").write_text(json.dumps(state))
    out = tmp_path / "handmade.pigeon"
    pack_run(rd, out)
    with zipfile.ZipFile(out) as z:
        assert json.loads(z.read("report.json"))["practice_set"]["source"] == \
            "drawn by hand and reviewed one by one"
        assert "drawn by hand and reviewed one by one" in z.read("README.md").decode()
        assert "drafted by a local LLM" not in z.read("README.md").decode()


def test_a_scorecard_says_which_runtime_measured_it(tmp_path):
    """A run folder assembled by hand can carry any `flight` block, and one
    measured in a runtime nobody serves with reads exactly like a real check ride
    until someone runs the file. Measured the hard way: an adapter that
    reproduced its corpus in MLX scored 0/20 through llama.cpp on the same
    questions."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    out = tmp_path / "unstamped.pigeon"
    pack_run(rd, out)
    with zipfile.ZipFile(out) as z:
        assert json.loads(z.read("report.json"))["check_ride"]["runtime"] == "unrecorded"

    stamped = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    state = json.loads((stamped / "state.json").read_text())
    state["flight"]["runtime"] = "llama.cpp"
    (stamped / "state.json").write_text(json.dumps(state))
    out2 = tmp_path / "stamped.pigeon"
    pack_run(stamped, out2)
    with zipfile.ZipFile(out2) as z:
        assert json.loads(z.read("report.json"))["check_ride"]["runtime"] == "llama.cpp"


# ---------------------------------------------------------------------------
# skills: a Pigeon whose answer is a RECIPE, and the data that renders it


def _skill_run(tmp_path):
    """A run whose adapter answers with a recipe rather than the drawing."""
    import facegrammar as fg
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False, flight=False)
    state = json.loads((rd / "state.json").read_text())
    state["skill"] = {"name": fg.SKILL_NAME, "version": fg.SKILL_VERSION,
                      "fields": list(fg.RECIPE_KEYS), "payload": fg.skill_payload()}
    (rd / "state.json").write_text(json.dumps(state))
    return rd


def test_a_skill_pigeon_names_its_skill_and_carries_it_as_data(tmp_path):
    out = tmp_path / "skilled.pigeon"
    m = pack_run(_skill_run(tmp_path), out)
    assert m["skill"] == {"name": "face-grammar", "version": 1}
    assert m["output_contract"]["type"] == "json"
    assert m["output_contract"]["fields"] == ["eyes", "mouth", "brows", "extras"]
    with zipfile.ZipFile(out) as z:
        payload = json.loads(z.read("skill/skill.json"))
    # Renderable from the payload ALONE, the way a foreign app would: no code
    # ships in a .pigeon, so rendering must be concatenation.
    recipe = {"eyes": "happy", "mouth": "grin", "brows": None, "extras": ["halo"]}
    frags = [payload["head"]]
    for key in payload["order"]:
        if key == "extras":
            frags += [payload["parts"]["extras"][x] for x in recipe["extras"]]
        elif recipe.get(key):
            frags.append(payload["parts"][key][recipe[key]])
    import facegrammar as fg
    assert payload["wrapper"].replace("{parts}", "".join(frags)) == fg.render(recipe)
    assert validate_pigeon(out, bases=installed_bases([tmp_path])).ok


def test_a_skill_is_accepted_by_how_it_is_drawn_not_by_its_name(tmp_path):
    """A skill no app has heard of opens, because it says how it is drawn and
    carries its own table. A method this app cannot perform is refused by name,
    the way an unknown `kind` is — drawing with the wrong one makes a well-formed
    picture of the wrong thing."""
    out = tmp_path / "s.pigeon"; pack_run(_skill_run(tmp_path), out)
    bases = installed_bases([tmp_path])

    def renamed_manifest(m):
        m["skill"]["name"] = "chart-grammar"
        return m
    renamed = _rewrite(_repacked(out, tmp_path / "chart-m.pigeon", renamed_manifest),
                       tmp_path / "chart.pigeon", "skill/skill.json",
                       lambda b: json.dumps({**json.loads(b), "name": "chart-grammar"}).encode())
    assert validate_pigeon(renamed, bases=bases).ok

    later = _rewrite(renamed, tmp_path / "layers.pigeon", "skill/skill.json",
                     lambda b: json.dumps({**json.loads(b), "render": {"method": "layers", "version": 1}}).encode())
    r = validate_pigeon(later, bases=bases)
    assert not r.ok and any("'layers' v1" in e and "update the app" in e for e in r.errors), r.errors

    unsaid = _rewrite(renamed, tmp_path / "unsaid.pigeon", "skill/skill.json",
                      lambda b: json.dumps({k: v for k, v in json.loads(b).items() if k != "render"}).encode())
    r = validate_pigeon(unsaid, bases=bases)
    assert not r.ok and any("does not say how 'chart-grammar' is drawn" in e for e in r.errors), r.errors


def test_a_face_grammar_pigeon_packed_before_render_existed_still_opens(tmp_path):
    """emoji-designer.pigeon shipped without `render`; it is read as concat v1."""
    out = tmp_path / "s.pigeon"; pack_run(_skill_run(tmp_path), out)
    legacy = _rewrite(out, tmp_path / "legacy.pigeon", "skill/skill.json",
                      lambda b: json.dumps({k: v for k, v in json.loads(b).items() if k != "render"}).encode())
    assert validate_pigeon(legacy, bases=installed_bases([tmp_path])).ok


def test_a_skill_pigeon_without_its_payload_is_rejected(tmp_path):
    out = tmp_path / "s2.pigeon"; pack_run(_skill_run(tmp_path), out)
    stripped = tmp_path / "hollow-skill.pigeon"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(stripped, "w", zipfile.ZIP_STORED) as dst:
        for n in src.namelist():
            if n != "skill/skill.json":
                dst.writestr(n, src.read(n))
    r = validate_pigeon(stripped, bases=installed_bases([tmp_path]))
    assert not r.ok and any("skill/skill.json" in e for e in r.errors)


# ---------------------------------------------------------------------------
# parity with Pigeon's reader (the reader brief)
#
# The goal is that Flight School refuses exactly what Pigeon refuses, so a file
# this app accepts is a file Pigeon opens. Each case below is a place the two
# readers used to disagree, and each disagreement was silent until someone
# tried to import the file on the other side.

from pigeon import nest as nest_mod  # noqa: E402
from pigeon import seal as seal_mod  # noqa: E402
from pigeon import store as store_mod  # noqa: E402


def _rewrite(src: Path, dst: Path, member: str, edit, method=zipfile.ZIP_STORED) -> Path:
    """Copy a .pigeon, passing one member's bytes through `edit`."""
    with zipfile.ZipFile(src) as z, zipfile.ZipFile(dst, "w", method) as out:
        for n in z.namelist():
            data = z.read(n)
            out.writestr(n, edit(data) if n == member else data)
    return dst


def _plain(tmp_path, name="plain.pigeon") -> Path:
    out = tmp_path / name
    pack_run(run_folder(tmp_path, base_q4(tmp_path), labels=False), out)
    return out


def test_the_strict_zip_rules_apply_to_an_unsealed_file_too(tmp_path):
    """Pigeon applies them to every import. A deflated re-zip that zipfile reads
    happily used to pass here and fail there."""
    src = _plain(tmp_path)
    deflated = _rewrite(src, tmp_path / "deflated.pigeon", "", lambda b: b, zipfile.ZIP_DEFLATED)
    assert validate_pigeon(src, bases=installed_bases([tmp_path])).ok
    r = validate_pigeon(deflated, bases=installed_bases([tmp_path]))
    assert not r.ok and any("refused" in e for e in r.errors)


def test_a_base_with_no_name_or_size_matches_no_other_file(tmp_path):
    """An empty name or size matches nothing by identity (§7.2): such a Pigeon is
    refused on any other file of its base, and its refusal leaves the empty
    field out of `needs` (D63). It reaches that step; it is not refused early."""
    src = _plain(tmp_path)
    for key in ("name", "size"):
        def blank(m, key=key):
            m["base_model"][key] = ""
            m["base_model"]["file"] = "some-other-file.gguf"
            return m
        r = validate_pigeon(_repacked(src, tmp_path / f"no-{key}.pigeon", blank),
                            bases=installed_bases([tmp_path]))
        assert not r.ok and r.code == "wrong_base", r.errors
        assert key not in r.needs["identity"]


def test_another_version_of_a_known_skill_is_refused(tmp_path):
    """The parts table is the contract; face-grammar v2 is not face-grammar v1."""
    out = tmp_path / "s.pigeon"; pack_run(_skill_run(tmp_path), out)

    def v2(m):
        m["skill"]["version"] = 2
        return m
    r = validate_pigeon(_repacked(out, tmp_path / "v2.pigeon", v2), bases=installed_bases([tmp_path]))
    assert not r.ok and any("v2" in e for e in r.errors)


def test_a_skill_payload_is_checked_not_only_present(tmp_path):
    out = tmp_path / "s.pigeon"; pack_run(_skill_run(tmp_path), out)
    bases = installed_bases([tmp_path])

    def no_order(b):
        p = json.loads(b); del p["order"]; return json.dumps(p).encode()

    def wrong_version(b):
        p = json.loads(b); p["version"] = "2"; return json.dumps(p).encode()

    def with_(**changes):
        return lambda b: json.dumps({**json.loads(b), **changes}).encode()

    def no_table(b):
        p = json.loads(b); del p["parts"]["mouth"]; return json.dumps(p).encode()

    for label, edit, needle in (("no-order", no_order, "order"),
                                ("wrong-version", wrong_version, "table it was trained against"),
                                ("not-json", lambda b: b"{nope", "could not be read"),
                                ("two-slots", with_(wrapper="{parts}{parts}"), "exactly once"),
                                ("no-table", no_table, "no parts table for 'mouth'"),
                                ("concat-v2", with_(render={"method": "concat", "version": 2}), "'concat' v2")):
        r = validate_pigeon(_rewrite(out, tmp_path / f"{label}.pigeon", "skill/skill.json", edit), bases=bases)
        assert not r.ok and any(needle in e for e in r.errors), (label, r.errors)
    # A payload version written as a numeric string is the same version.
    ok = _rewrite(out, tmp_path / "string-version.pigeon", "skill/skill.json",
                  lambda b: json.dumps({**json.loads(b), "version": "1"}).encode())
    assert validate_pigeon(ok, bases=bases).ok


def test_material_is_parsed_not_only_present(tmp_path):
    """Pigeon parses chunks.json at import; a file whose sections are empty or
    mistyped installed here and failed there."""
    rd = run_folder(tmp_path, base_q4(tmp_path), labels=False)
    add_material(rd)
    out = tmp_path / "g.pigeon"; pack_run(rd, out)
    bases = installed_bases([tmp_path])
    assert validate_pigeon(out, bases=bases).ok
    cases = {
        "empty": ({"chunks": []}, "no sections"),
        "no-text": ({"chunks": [{"ordinal": 1, "title": "t"}]}, "'text'"),
        "float-ordinal": ({"chunks": [{"ordinal": 1.5, "text": "x"}]}, "'ordinal'"),
        "null-title": ({"chunks": [{"ordinal": 1, "text": "x", "title": None}]}, "'title'"),
    }
    for label, (body, needle) in cases.items():
        r = validate_pigeon(_rewrite(out, tmp_path / f"{label}.pigeon", "material/chunks.json",
                                     lambda b, body=body: json.dumps(body).encode()), bases=bases)
        assert not r.ok and any(needle in e for e in r.errors), (label, r.errors)


def test_nest_members_are_named_by_id_never_by_file_name(tmp_path):
    """`Coin Sorter.pigeon` or `Café.pigeon` as a member name fails Pigeon's strict
    rules for the CONTAINER, which refuses the whole nest."""
    src = _plain(tmp_path)
    a = tmp_path / "Coin Sorter.pigeon"; a.write_bytes(src.read_bytes())
    b = tmp_path / "Café.PIGEON"; b.write_bytes(src.read_bytes())
    listing = nest_mod.pack(tmp_path / "flock.nest", [a, b])
    names = [p["file"] for p in listing["pigeons"]]
    assert names == ["coin-sorter.pigeon", "coin-sorter-2.pigeon"]
    seal_mod.strict_entries(tmp_path / "flock.nest")          # the container passes
    assert nest_mod.list_nest(tmp_path / "flock.nest")["members"] == names


def _skillpack_bytes(tmp_path) -> bytes:
    p = tmp_path / "patch-src.pigeon"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
        z.writestr("pigeon.json", json.dumps({"format_version": 1, "id": "patch", "kind": "skillpack"}))
    return p.read_bytes()


def _sealed_pack(tmp_path, needs) -> Path:
    """A skill pack sealed with the spec's test key (RFC 8032 TEST 1)."""
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    body = {"pigeon.json": json.dumps({"format_version": 1, "name": "Patch", "id": "patch", "kind": "skillpack",
                                       "version": 1, "models": {}, "needs": needs,
                                       "code": {"entry": "code/skills/index.js"}}).encode(),
            "code/skills/index.js": b"export default [];\n"}
    sig = seal_mod.sign(seed, seal_mod.seal_message(sorted(body.items())))
    key = seal_mod.key_id(seal_mod.public_key(seed))
    p = tmp_path / f"patch-{len(needs)}.pigeon"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
        for name, data in body.items():
            z.writestr(name, data)
        z.writestr(seal_mod.SEAL_NAME, json.dumps({"scheme": "pigeon-seal-ed25519-v1", "key": key,
                                                   "signature": sig.hex()}, separators=(",", ":")))
    return p


def test_a_sealed_skillpack_travels_beside_the_pigeons_it_needs(tmp_path):
    """D84: Patch ships as his sealed pack beside the Pigeons it asks. Pigeon
    opens that nest only with one pack and every Pigeon it names."""
    coins = _plain(tmp_path)
    coin_id = json.loads(zipfile.ZipFile(coins).read("pigeon.json"))["id"]
    listing = nest_mod.pack(tmp_path / "patch.nest", [_sealed_pack(tmp_path, [coin_id]), coins])
    assert [p["kind"] for p in listing["pigeons"]] == ["skillpack", "freeform"]
    seal_mod.strict_entries(tmp_path / "patch.nest")
    results = store_mod.add(tmp_path / "patch.nest", tmp_path / "store", [tmp_path])
    by = {Path(r["file"]).name: r for r in results}
    assert by["patch.pigeon"]["status"] == "refused" and by["patch.pigeon"]["errors"] == [nest_mod.SKILLPACK_IN_NEST]
    assert by[f"{coin_id}.pigeon"]["status"] == "added", "the rest are kept (§14)"

    with pytest.raises(ValueError, match="needs pen, which this nest does not carry"):
        nest_mod.pack(tmp_path / "short.nest", [_sealed_pack(tmp_path, [coin_id, "pen"]), coins])
    sp = tmp_path / "unsealed.pigeon"; sp.write_bytes(_skillpack_bytes(tmp_path))
    with pytest.raises(ValueError, match="travels sealed"):
        nest_mod.pack(tmp_path / "bad.nest", [_plain(tmp_path), sp])
    with pytest.raises(ValueError, match="one skill pack at most"):
        nest_mod.pack(tmp_path / "two.nest", [_sealed_pack(tmp_path, []), _sealed_pack(tmp_path, [coin_id]), coins])
    assert not any((tmp_path / n).exists() for n in ("short.nest", "bad.nest", "two.nest"))


def _hand_nest(path: Path, members: dict) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr("nest.json", json.dumps({"format_version": 1, "name": "hand", "pigeons": []}))
    return path


def test_a_skillpack_member_is_refused_but_the_good_ones_land(tmp_path):
    good = _plain(tmp_path).read_bytes()
    nest = _hand_nest(tmp_path / "mixed.nest", {"good.pigeon": good,
                                                "patch.pigeon": _skillpack_bytes(tmp_path)})
    results = store_mod.add(nest, tmp_path / "store", [tmp_path])
    by = {Path(r["file"]).name: r for r in results}
    assert by["good.pigeon"]["status"] == "added"
    assert by["patch.pigeon"]["status"] == "refused"
    assert by["patch.pigeon"]["errors"] == [nest_mod.SKILLPACK_IN_NEST]


def test_an_uppercase_extension_is_not_a_member(tmp_path):
    """Pigeon's match is case-sensitive; a nest of only `.PIGEON` members has none."""
    nest = _hand_nest(tmp_path / "loud.nest", {"Loud.PIGEON": _plain(tmp_path).read_bytes()})
    with pytest.raises(nest_mod.NotANest, match="no .pigeon files"):
        nest_mod.list_nest(nest)


def test_a_bad_member_name_refuses_the_whole_nest(tmp_path):
    nest = _hand_nest(tmp_path / "spaced.nest", {"good.pigeon": _plain(tmp_path).read_bytes(),
                                                 "Coin Sorter.pigeon": _plain(tmp_path, "b.pigeon").read_bytes()})
    with pytest.raises(nest_mod.NotANest, match="refused"):
        nest_mod.list_nest(nest)


# ---------------------------------------------------------------------------
# The check ride is a Runner (spec §8): its score must be one a host reproduces


class _RideServer:
    """Answers from a script and keeps every user turn it was sent."""
    class spec:
        n_parallel = 1

    def __init__(self, answers):
        self.answers, self.sent = answers, []

    def chat(self, messages, **kw):
        self.sent.append((messages[-1]["content"], kw.get("grammar")))
        self.max_tokens = kw.get("max_tokens")
        return self.answers[len(self.sent) - 1] if len(self.sent) <= len(self.answers) else ""


def test_the_ride_stops_where_the_file_says_a_runner_stops(tmp_path):
    """§5.4, §8.5: a Runner MUST cap generation at output_contract.max_tokens,
    so the ride, which is a Runner, stops at the number the packer writes.
    pat-v2's ride once ran to twice its answer budget, past the file's 172."""
    import json as _json
    from instructor.fly import answer_budget, run_flight
    from pigeon.manifest import answer_cap
    (tmp_path / "export").mkdir()
    (tmp_path / "measured.json").write_text(_json.dumps({"answer_tokens_measured": 115}))
    adapter = tmp_path / "export" / "a.gguf"
    prose = {"output_shape": "a paragraph", "max_answer_tokens": 80, "constrained": False}
    assert answer_budget(adapter, prose) == answer_cap(prose, 115, grounded=False) == 172
    labels = {"output_shape": "exactly one of: KEY(Up), KEY(Down)", "max_answer_tokens": 5}
    (tmp_path / "measured.json").write_text(_json.dumps({"answer_tokens_measured": 4}))
    assert answer_budget(adapter, labels) == answer_cap(labels, 4, grounded=False) == 8
    srv = _RideServer(["a"])
    run_flight(srv, prose, [{"question": "q", "answer": "a"}], max_answer_tokens=172, judge=False, control=False)
    assert srv.max_tokens == 172


def test_the_ride_reads_labels_as_a_runner_does():
    """§8.5: case-folded, trimmed, never repaired. "KEY(Up)." used to pass the
    ride (all punctuation stripped) and come back off_contract in every host."""
    from instructor.fly import run_flight
    plan = {"output_shape": "exactly one of: KEY(Up), KEY(Down)", "constrained": False}
    rows = [{"question": "q", "answer": "KEY(Up)"}] * 4
    srv = _RideServer(["key(up)", " KEY(Up) ", "KEY(Up).", "KEY(Down)"])
    f = run_flight(srv, plan, rows, max_answer_tokens=8, judge=False, control=False)
    assert [r.passed for r in f.rows] == [True, True, False, False]


def test_the_ride_reads_the_last_do_of_a_reasoned_answer():
    """§8.5 `reasoned`: the last <do>…</do>; an unclosed last <do> is
    off_contract even when an earlier one closed."""
    from instructor.rules_table import reasoned
    want = "<think>go up</think>\n<do>\nKEY(Up)\n</do>"
    assert reasoned(want, "<think>x</think><do>KEY(Down)</do> no <do>KEY(Up)</do>")[0]
    assert not reasoned(want, "<do>KEY(Up)</do> then <do>KEY(Up)")[0]
    assert not reasoned(want, "KEY(Up)")[0]
    assert not reasoned(want, "<DO>KEY(Up)</DO>")[0], "the tag is literal"


def test_the_ride_fills_the_template_as_a_runner_does(tmp_path):
    """§8.2: the input trimmed and filled in one pass; a Reader's section block
    above it, built from the chunks, not taken from the row as stored."""
    from instructor.fly import run_flight
    plan = {"output_shape": "one tidy sentence", "constrained": False}
    srv = _RideServer(["a"])
    run_flight(srv, plan, [{"question": "  hello {input} \n", "answer": "a"}], max_answer_tokens=8,
               judge=False, control=False)
    assert srv.sent[0][0] == "hello {input}"
    chunk = {"ordinal": 4, "title": "Memory", "text": "Needs 3.8 GB."}
    srv = _RideServer(["b"])
    run_flight(srv, {**plan, "grounded": True}, [{"question": "[§4] Memory\nNeeds 3.8 GB.\n\n  how much? ", "answer": "b"}],
               max_answer_tokens=8, judge=False, control=False, chunks=[chunk])
    assert srv.sent[0][0] == "[§4] Memory\nNeeds 3.8 GB.\n\nhow much?"


def test_label_rides_are_constrained_and_reasoned_rides_are_not():
    """D27 / D36: a label answer is generated inside the label set; a reasoned
    answer is left free, because constraining its <do> is unmeasured."""
    from instructor.fly import run_flight
    srv = _RideServer(["KEY(Up)"])
    f = run_flight(srv, {"output_shape": "exactly one of: KEY(Up), KEY(Down)"}, [{"question": "q", "answer": "KEY(Up)"}],
                   max_answer_tokens=8, judge=False, control=False)
    assert f.constrained and srv.sent[0][1] == 'root ::= "KEY(Up)" | "KEY(Down)"'
    srv = _RideServer(["<do>KEY(Up)</do>"])
    f = run_flight(srv, {"output_shape": "the answer", "grader": {"how": "rules", "rule": "instructor.rules_table:reasoned"}},
                   [{"question": "q", "answer": "<do>KEY(Up)</do>"}], max_answer_tokens=8, judge=False, control=False)
    assert not f.constrained and srv.sent[0][1] is None and f.rows[0].passed


def test_serving_that_trails_training_is_named_and_noise_is_not():
    """§12.1: the MLX ride exists to catch an adapter that scores where it
    trained and loses it when served. Measured: 4-bit training, 169 in MLX and
    139 served; bf16, 167 and 164. The bar is the bake-off's noise floor."""
    from pigeon.pack import serving_trails
    assert "30 answers are lost" in serving_trails({"passed": 139, "total": 200}, {"passed": 169})
    assert serving_trails({"passed": 164, "total": 200}, {"passed": 167}) is None
    assert serving_trails({"passed": 20, "total": 30}, {"passed": 25}) is None, "5 of 30 is one base's seed-to-seed spread"


def test_both_rides_travel_in_check_rides(tmp_path):
    """D21/D30: one entry per ride; the llama.cpp ride names the file it ran
    on, the MLX ride the precision it ran at."""
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base)
    st = json.loads((rd / "state.json").read_text())
    st["flight"]["check_rides"] = [
        {"runtime": "llama.cpp", "passed": 5, "total": 6, "control_passed": 1, "temperature": 0, "constrained": True},
        {"runtime": "mlx", "passed": 6, "total": 6, "control_passed": None, "temperature": 0, "constrained": False,
         "precision": "bf16"}]
    (rd / "state.json").write_text(json.dumps(st))
    out = tmp_path / "t.pigeon"
    pack_run(rd, out, prompt_check_builder=fixed_prompt_check)
    with zipfile.ZipFile(out) as z:
        rides = json.loads(z.read("report.json"))["check_rides"]
    assert [r["runtime"] for r in rides] == ["llama.cpp", "mlx"]
    assert rides[0]["file_sha256"] == sha(base) and "file_sha256" not in rides[1] and rides[1]["precision"] == "bf16"


def test_a_readers_same_source_score_travels_in_the_report(tmp_path):
    """The deterministic Reader check reaches report.json beside the judged
    score, and each row keeps its own verdict."""
    rd = run_folder(tmp_path, base_q4(tmp_path))
    st = json.loads((rd / "state.json").read_text())
    st["flight"]["same_source"] = {"same": 4, "n": 6, "base_same": 1, "base_n": 6}
    st["flight"]["rows"][0]["same_source"] = {"same": True, "expected": "E", "got": "E"}
    (rd / "state.json").write_text(json.dumps(st))
    out = tmp_path / "t.pigeon"
    pack_run(rd, out, prompt_check_builder=fixed_prompt_check)
    with zipfile.ZipFile(out) as z:
        ride = json.loads(z.read("report.json"))["check_ride"]
    assert ride["same_source"] == {"same": 4, "n": 6, "base_same": 1, "base_n": 6}
    assert ride["rows"][0]["same_source"]["same"] is True and "same_source" not in ride["rows"][1]


# ---------------------------------------------------------------------------
# base Pigeons (D83): a base with a system prompt, packed with no training

CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


class _FakeRider:
    """Stands in for `pigeon-ride`: answers from a script, read the way the
    engine reads them (the kit's reader), and keeps what it was given."""
    def __init__(self, answers, control=None, engine="libpigeon test (llama.cpp abc; spec 1)"):
        self.answers, self.control, self.engine = list(answers), control, engine

    def _events(self, answers, m, labels):
        from pigeon.runner import read_answer
        return [{"type": "answer", "raw": raw, "stats": {"prompt_tokens": 9, "total_ms": 4},
                 "answer": read_answer(m["output_contract"], raw, kind=m["kind"],
                                       labels=(row.get("host_values") or {}).get("label") or labels)}
                for row, raw in zip(self.rows, answers)]

    def __call__(self, path, rows):
        z = zipfile.ZipFile(path)
        self.path, self.names, self.rows = path, z.namelist(), rows
        m = json.loads(z.read("pigeon.json"))
        labels = z.read("labels.txt").decode().split() if "labels.txt" in self.names else None
        return {"adapter": {"events": self._events(self.answers, m, labels)},
                "control": {"events": self._events(self.control, m, labels)} if self.control else None,
                "engine": self.engine}


def _base_table(tmp_path, table: dict, rows: list[dict]) -> Path:
    from pigeon import basepack
    d = tmp_path / "table"; d.mkdir()
    gguf = write_gguf(tmp_path / "Base-Q4_K_M.gguf", {**QWEN25, "general.file_type": 15,
                                                      "tokenizer.chat_template": CHATML})
    (d / basepack.TABLE_FILE).write_text(json.dumps({"base_gguf": str(gguf), "rows_source": "a test", **table}))
    (d / basepack.ROWS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_a_base_pigeon_packs_with_no_adapter_and_rides_through_the_engine(tmp_path):
    """D83, D85, D86, D98: no adapter; the file the engine rode carries no
    report yet; the report names the engine that answered."""
    from pigeon import basepack
    shots = [{"user": "round 12.7", "pigeon": "YES"}]
    d = _base_table(tmp_path, {"name": "Asks", "id": "patch-asks", "kind": "classifier",
                               "system_prompt": "YES or NO.", "shots": shots, "labels": ["YES", "NO"],
                               "prompt_template": "Message: {input}",
                               "output_contract": {"type": "label", "max_tokens": 6, "constrained": True},
                               "card": {"does": "Says whether a message asks for something."}},
                    [{"input": "split 40 three ways", "expected": "YES"}, {"input": "nice weather", "expected": "YES"}])
    rider = _FakeRider(["yes", "NO"])
    got = basepack.pack_base(d, tmp_path / "asks.pigeon", cfg=None, rider=rider, icon_png=b"\x89PNG")
    m = got["manifest"]
    assert m["base_only"] is True and got["flight"]["passed"] == 1 and got["flight"]["total"] == 2, \
        "'yes' reads as YES (§8.5 folds ASCII case); NO is wrong"
    assert got["flight"]["off_contract"] == 0 and got["flight"]["control_passed"] is None
    assert "report.json" not in rider.names and "pigeon.json" in rider.names
    assert not rider.path.exists(), "the copy the engine rode is not left behind"
    assert m["prompt_check"]["canonical"].startswith("<|im_start|>system\nYES or NO.<|im_end|>\n<|im_start|>user\nround 12.7")
    names = zipfile.ZipFile(tmp_path / "asks.pigeon").namelist()
    assert "adapter.lora.gguf" not in names and {"shots.json", "system_prompt.txt", "labels.txt"} <= set(names)
    rep = json.loads(zipfile.ZipFile(tmp_path / "asks.pigeon").read("report.json"))
    assert rep["check_ride"]["rows_source"] == "a test" and m["card"]["check"] == {"passed": 1, "total": 2}
    assert rep["check_ride"]["engine"] == rider.engine and rep["check_rides"][0]["engine"] == rider.engine
    assert rep["check_ride"]["rows"][0]["ms"] == 4
    r = validate_pigeon(tmp_path / "asks.pigeon", bases=installed_bases([tmp_path]), general_reader=True)
    assert r.ok, r.errors


def test_a_base_pigeons_host_values_hold_its_json_and_prose_is_a_sample(tmp_path):
    """D88: a row's host values travel to the engine as they are. A text
    answer is recorded with no score (§12.2)."""
    from pigeon import basepack
    schema = {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]}
    d = _base_table(tmp_path, {"name": "Second look", "kind": "extractor", "prompt_template": "{input}",
                               "output_contract": {"type": "json", "max_tokens": 80, "constrained": True,
                                                   "host_values": ["skill"], "output_schema": schema}},
                    [{"input": "add 2 and 3", "expected": {"skill": "sum"}, "host_values": {"skill": ["sum", "none"]}},
                     {"input": "a poem", "host_values": {"skill": ["sum", "none"]}}])
    rider = _FakeRider(['{"skill":"sum"}', '{"skill":"none"}'])
    got = basepack.pack_base(d, tmp_path / "look.pigeon", cfg=None, rider=rider, icon_png=b"\x89PNG")
    assert rider.rows[0]["host_values"] == {"skill": ["sum", "none"]}
    assert (got["flight"]["passed"], got["flight"]["total"], got["flight"]["samples"]) == (1, 1, 1)
    assert basepack.field_value({"steps": [{"skill": "a"}, {"skill": "b"}]}, "steps[].skill") == ["a", "b"]
    assert basepack.field_value({"skill": "sum"}, "skill") == "sum"
    with pytest.raises(ValueError, match="host values name a field"):
        basepack.ask_kwargs(got["manifest"], None, {"nope": ["x"]})

    (tmp_path / "pen").mkdir()
    d = _base_table(tmp_path / "pen", {"name": "Pen", "kind": "persona", "system_prompt": "Write plainly.",
                                       "prompt_template": "{input}", "temperature": 0.7,
                                       "history": {"shape": "as_sent", "max_turns": 12},
                                       "output_contract": {"type": "text", "max_tokens": 400}},
                    [{"input": "and tomorrow?", "turns": ["what should I cook tonight"]}])
    rider = _FakeRider(["Soup."])
    got = basepack.pack_base(d, tmp_path / "pen.pigeon", cfg=None, rider=rider, icon_png=b"\x89PNG")
    assert got["flight"]["total"] is None and got["flight"]["samples"] == 1
    assert rider.rows[0]["turns"] == ["what should I cook tonight"], "the turns go to the engine ride (engine_ride.ask)"


def test_a_trained_pigeon_from_elsewhere_carries_its_adapter_and_a_control(tmp_path, monkeypatch):
    """Patch's router: trained with a system prompt (D85) outside Flight
    School. The engine rides the file and its D83 control, and prompt_check
    comes from the training renderer, not the Runner's."""
    from pigeon import basepack
    lora = write_gguf(tmp_path / "router.lora.gguf", {"general.architecture": "qwen2", "adapter.type": "lora"}, "adapter")
    d = _base_table(tmp_path, {"name": "Router", "id": "patch-router", "kind": "router", "system_prompt": "Route it.",
                               "labels": ["math", "none"], "prompt_template": "{input}", "adapter": str(lora),
                               "trained_from": {"repo": "mlx-community/x", "precision": "bf16"},
                               "output_contract": {"type": "label", "max_tokens": 12, "constrained": True}},
                    [{"input": "add 2 and 3", "expected": "math"}, {"input": "hello", "expected": "none"}])
    monkeypatch.setattr(basepack, "trained_prompt_check",
                        lambda t, row: {"input": row["input"], "canonical": "<|im_start|>system\nRoute it.<|im_end|>\n"
                                        "<|im_start|>user\nadd 2 and 3<|im_end|>\n<|im_start|>assistant\n"})
    rider = _FakeRider(["math", "none"], control=["none", "none"])
    got = basepack.pack_base(d, tmp_path / "router.pigeon", cfg=None, rider=rider, icon_png=b"\x89PNG")
    assert "adapter.lora.gguf" in rider.names and "base_only" not in got["manifest"]
    assert (got["flight"]["passed"], got["flight"]["control_passed"]) == (2, 1)
    z = zipfile.ZipFile(tmp_path / "router.pigeon")
    assert "adapter.lora.gguf" in z.namelist()
    assert json.loads(z.read("report.json"))["check_ride"]["control_passed"] == 1
    assert validate_pigeon(tmp_path / "router.pigeon", bases=installed_bases([tmp_path]), general_reader=True).ok


def test_a_plan_is_graded_on_its_steps_not_its_about():
    from instructor.rules_table import plan_steps
    want = '{"about":"add","steps":[{"id":"s1","skill":"sum","input":{"values":[2,3]}}]}'
    assert plan_steps(want, '{"about":"adding two","steps":[{"id":"s1","skill":"sum","input":{"values":[2,3]}}]}')[0]
    assert plan_steps(want, '{"about":"add","steps":[{"id":"s1","skill":"sum","input":{"values":[2]}}]}') == \
        (False, "the right skills, other inputs")
    assert not plan_steps(want, 'steps: sum')[0]
    assert plan_steps('{"about":"x","steps":[]}', '{"about":"y","steps":[]}')[0], "a decline is a plan too"


def test_a_trained_pigeons_prompt_check_renders_the_filled_template(tmp_path, monkeypatch):
    """The training renderer is asked for the user turn as the rows carried
    it: the input inside the template, trimmed, never the bare input."""
    import subprocess
    from pigeon import basepack
    seen = {}

    class Done:
        stdout = json.dumps({"canonical": "x"})

    def fake_run(cmd, **kw):
        seen.update(json.loads(cmd[-1]))
        return Done()
    monkeypatch.setattr(subprocess, "run", fake_run)
    t = {"trained_from": {"repo": "r"}, "train_python": "py", "prompt_template": 'Message: "{input}"\n\nYES or NO:',
         "system_prompt": "One word.", "shots": [{"user": "a", "pigeon": "YES"}]}
    pc = basepack.trained_prompt_check(t, {"input": "  he rounded on me "})
    assert seen["user"] == 'Message: "he rounded on me"\n\nYES or NO:'
    assert seen["system"] == "One word." and seen["shots"] == t["shots"]
    assert pc["input"] == "he rounded on me"
