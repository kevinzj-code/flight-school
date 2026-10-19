"""The parts of a Reader and Runner the conformance kit's second slice asks for:
retrieval in Python, the base registry, the Runner's prompt check, key presses
and a nest's refusal codes. The kit holds the full cases (bench/test_pigeon.py,
PIGEON_CONFORMANCE_KIT); these run without it.

    python3 -m pytest src-tauri/python/bench/test_reader_parts.py -q
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import nest, promptproof, registry, retrieve, runner, seal  # noqa: E402

from bench import make_seal_fixture as fixture  # noqa: E402

TEST_KEY = seal.key_id(fixture.TEST_PUB)
TRUSTED = {TEST_KEY: fixture.TEST_PUB.hex()}

CHUNKS = [{"ordinal": 1, "title": "Memory", "text": "A model needs about 3.8 GB of free memory."},
          {"ordinal": 2, "title": "Batteries", "text": "Replace the batteries every six months."},
          {"ordinal": 3, "title": "Fifteenth amendment", "text": "Voting rights shall not be denied by race."}]


def test_retrieval_picks_one_section_or_none():
    assert retrieve.pick("whole", CHUNKS, "anything") == [1, 2, 3]
    assert retrieve.pick("keyword-overlap-v1", CHUNKS, "When to replace batteries?") == [2]
    assert retrieve.pick("keyword-overlap-v1", CHUNKS, "Any memory?") == []          # 6 bytes matched, floor is 9
    assert retrieve.pick("keyword-overlap-v2", CHUNKS, "What does the fifteenth amendment say?") == [3]
    assert retrieve.pick("keyword-overlap-v2", CHUNKS, "Who won the cup final?") == []
    assert retrieve.v2_tokens("The Twenty-Fourth and fifteenth parties' duties") == ["xxiv", "xv", "party", "duty"]
    units = [(0, [1, 0, 0, 0]), (1, [0, 1, 0, 0]), (2, [0, 0, 1, 0])]
    assert retrieve.pick("hybrid-v1", CHUNKS, "batteries", units=units, qvec=[0, 1, 0, 0]) == [2]
    assert retrieve.pick("hybrid-v1", CHUNKS, "batteries", units=units, qvec=[0, 0.5, 0, 0.866]) == []
    with pytest.raises(ValueError):
        retrieve.pick("bm25-v9", CHUNKS, "x")


REGISTRY = {"registry_version": 1, "bases": [{
    "id": "qwen3-vl-2b-instruct", "identity": {"arch": "qwen3vl", "name": "qwen3vl", "size": "2B", "finetune": "instruct"},
    "license": "Apache-2.0",
    "files": [{"file": "q8.gguf", "sha256": "8" * 64}, {"file": "q4.gguf", "sha256": "4" * 64, "recommended": True}],
    "projectors": [{"file": "mmproj.gguf", "sha256": "f" * 64, "recommended": True}]}]}


def test_a_registry_is_used_only_under_a_registry_key():
    entries = [("registry.json", json.dumps(REGISTRY).encode())]
    blob = seal.build_sealed(entries, fixture.TEST_SEED)
    reg, key = registry.open_registry(blob, TRUSTED)
    assert key == TEST_KEY and reg == REGISTRY
    want = {"arch": "qwen3vl", "name": "Qwen3Vl 2b Instruct", "size": "2B", "finetune": "instruct"}
    e = registry.entry_for(reg, want)
    assert e["id"] == "qwen3-vl-2b-instruct" and registry.recommended(e["files"])["file"] == "q4.gguf"
    assert registry.projector_hashes(e) == ["f" * 64]
    # D63: the entry has no license_url and the file row no bytes, so both are left out
    assert registry.needs(want, "q4.gguf", e) == {"file": "q4.gguf", "identity": want,
                                                 "registry_id": "qwen3-vl-2b-instruct", "license": "Apache-2.0"}
    assert registry.needs(want, "q4.gguf", None) == {"file": "q4.gguf", "identity": want}
    # D45: a spelling the entry accepts is the same base, from either side
    spelt = {**e, "names": ["Q3-VL"]}
    assert registry.header_is_entry(spelt, {**want, "name": "Q3-VL"})
    assert registry.entry_for({"bases": [spelt]}, {**want, "name": "Q3-VL"}) is spelt
    assert not registry.header_is_entry(e, {**want, "name": "Q3-VL"})
    # D46: a listed file whose own header is another model
    wrong = {"file": "q8.gguf", **want, "finetune": "thinking"}
    assert registry.disagrees(e, [wrong], lambda b: "8" * 64)
    assert not registry.disagrees(e, [wrong], lambda b: "0" * 64)
    # D61: a hash already known counts whatever the name; an unknown one is only
    # looked for under a listed name
    renamed = {**wrong, "file": "renamed.gguf"}
    assert registry.disagrees(e, [{**renamed, "sha256": "8" * 64}], lambda b: 1 / 0)
    assert not registry.disagrees(e, [renamed], lambda b: 1 / 0)
    assert registry.entry_for(reg, {**want, "finetune": "thinking"}) is None
    # One key, one purpose: no registry key, no registry — whatever else the host trusts.
    with pytest.raises(registry.RegistryRefused) as no_key:
        registry.open_registry(blob, {})
    assert no_key.value.code == "untrusted_seal"
    tampered = seal.strict_entries(blob)
    tampered[0] = ("registry.json", tampered[0][1] + b" ")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for n, d in tampered:
            z.writestr(seal._info(n, (1980, 1, 1, 0, 0, 0)), d)
    with pytest.raises(registry.RegistryRefused) as broken:
        registry.open_registry(out.getvalue(), TRUSTED)
    assert broken.value.code == "broken_seal"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr(seal._info("registry.json", (1980, 1, 1, 0, 0, 0)), entries[0][1])
    with pytest.raises(registry.RegistryRefused, match="not sealed") as unsealed:
        registry.open_registry(out.getvalue(), TRUSTED)
    assert unsealed.value.code == "bad_registry"
    newer = seal.build_sealed([("registry.json", json.dumps({**REGISTRY, "registry_version": 2}).encode())], fixture.TEST_SEED)
    with pytest.raises(registry.RegistryRefused) as too_new:
        registry.open_registry(newer, TRUSTED)
    assert too_new.value.code == "newer_format"


CHATML = ("{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' }}"
          "{%- if m.content is string %}{{- m.content }}{%- else %}{%- for c in m.content %}"
          "{%- if c.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{%- else %}{{- c.text }}{%- endif %}"
          "{%- endfor %}{%- endif %}{{- '<|im_end|>\\n' }}{%- endfor %}"
          "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")
PICTURE = {"markup": "<|vision_start|><|image_pad|><|vision_end|>", "tokens": 64}


def test_the_runner_proves_its_prompt_or_refuses():
    pytest.importorskip("jinja2")
    canonical = "<|im_start|>user\n<image:64>What do you do next?<|im_end|>\n<|im_start|>assistant\n"
    m = {"kind": "classifier", "inputs": ["image", "text"], "prompt_template": "{image}\n{input}",
         "prompt_check": {"input": {"text": "What do you do next?", "image": "examples/001.png"}, "canonical": canonical}}
    assert promptproof.check(m, CHATML, picture=PICTURE) == {"ok": True, "warnings": [], "rendered": canonical,
                                                            "position": None}
    other = promptproof.check(m, CHATML, picture={**PICTURE, "tokens": 256})
    assert not other["ok"] and other["position"] == len(canonical[:canonical.index("64>")].encode())
    del m["prompt_check"]
    assert promptproof.check(m, CHATML)["warnings"] == ["prompt_unchecked"]


def test_only_a_declared_key_is_pressed():
    raw = json.dumps({"format_version": 1, "keys": {"Up": {"code": 126, "hold_ms": 95}, "Down": {"code": 300}},
                      "not_pressed": [{"answer": "KEY(Save)", "why": "held with a modifier"}],
                      "frames": {"stack": 1, "gap_ms": 250, "max_side": 768}}).encode()
    actions = runner.usable_actions(raw, {"inputs": ["image", "text"]})
    label = lambda v: {"kind": "label", "value": v}  # noqa: E731
    assert runner.action_for(actions, label("KEY(Up)")) == {"action": {"key": "Up", "code": 126, "hold_ms": 95},
                                                           "not_pressed": None}
    assert runner.action_for(actions, label("KEY(Down)")) == {"action": None, "not_pressed": None}   # D68: code 300 is not a key
    assert runner.action_for(actions, label("KEY(Save)"))["not_pressed"]["why"] == "held with a modifier"
    assert runner.action_for(actions, {"kind": "off_contract", "raw": "KEY(Up)"})["action"] is None
    # D54: skipped when it is unreadable, or the Pigeon takes no pictures
    assert runner.usable_actions(raw, {"inputs": ["text"]}) is None
    assert runner.usable_actions(b"keys: Up", {"inputs": ["image", "text"]}) is None
    # D66, D67, D69: a format this app does not know, or frames it cannot follow
    pic = {"inputs": ["image", "text"]}
    for change in ({"format_version": 2}, {"format_version": None}, {"frames": None},
                   {"frames": {"stack": 1, "max_side": 32}}, {"frames": {"stack": 1, "max_side": 768, "crop": [0, 0, 1]}}):
        bad = {**json.loads(raw), **change}
        assert runner.usable_actions(json.dumps({k: v for k, v in bad.items() if v is not None}).encode(), pic) is None, change
    assert runner.action_for(None, label("KEY(Up)")) == {"action": None, "not_pressed": None}


def test_a_refused_nest_carries_the_spec_code(tmp_path):
    (tmp_path / "text.nest").write_text("hello")
    with pytest.raises(nest.NotANest) as e:
        nest.list_nest(tmp_path / "text.nest")
    assert e.value.code == "not_a_zip"
    with zipfile.ZipFile(tmp_path / "byhand.nest", "w") as z:
        z.writestr("one.pigeon", b"x")
    with pytest.raises(nest.NotANest) as e:
        nest.list_nest(tmp_path / "byhand.nest")
    assert e.value.code == "not_a_nest"
    with zipfile.ZipFile(tmp_path / "squeezed.nest", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("nest.json", json.dumps({"format_version": 1, "name": "x" * 200}))
        z.writestr("one.pigeon", b"x")
    with pytest.raises(nest.NotANest) as e:
        nest.list_nest(tmp_path / "squeezed.nest")
    assert e.value.code == "bad_zip"


def test_a_general_reader_refuses_a_sealed_skillpack_by_kind(tmp_path):
    from pigeon import validate_pigeon
    p = tmp_path / "f.pigeon"
    p.write_bytes(fixture.build()[0])
    assert validate_pigeon(p, trusted_keys=TRUSTED).ok                       # Flight School seals these, so it reads them
    r = validate_pigeon(p, trusted_keys=TRUSTED, general_reader=True)
    assert not r.ok and r.code == "unsupported_kind"
    assert validate_pigeon(p, general_reader=True).code == "untrusted_seal"  # the seal is judged first


def test_the_capture_plan_is_the_recorders():
    """§13.2 steps 0 to 3, against the numbers Recorder.swift's own self-test
    asserts (captureSize, cropped), and D70's trimming."""
    plan = lambda w, f: runner.frames_plan(w, {"stack": 1, "gap_ms": 250, **f})  # noqa: E731
    assert plan([3200, 2000], {"max_side": 768, "crop": [0, 0, 0.5, 0.5]})["capture"] == [1536, 960]
    assert plan([1600, 1000], {"max_side": 768, "crop": [0, 0, 0.25, 0.5]})["capture"] == [1600, 1000]
    # trimmed first: the crop past the edge keeps 320 x 200, so nothing shrinks
    assert plan([3200, 2000], {"max_side": 768, "crop": [0.9, 0, 0.5, 0.1]}) == \
        {"result": "followed", "capture": [3200, 2000], "kept": [2880, 0, 320, 200]}
    assert plan([1000, 1000], {"max_side": 2000, "crop": [-0.1, 0, 0.5, 1]})["kept"] == [0, 0, 500, 1000]
    # the longer side is always exactly max_side when the window shrinks
    for w, h in ((1920, 1200), (1366, 768), (2560, 1600), (3021, 1890)):
        kept = plan([w, h], {"max_side": 160, "crop": [0.242, 0.1509, 0.5171, 0.8228]})["kept"]
        assert max(kept[2:]) == 160, (w, h, kept)
    assert plan([6000, 300], {"max_side": 768, "crop": [0, 0, 1, 0.012]}) == {"result": "skipped"}
    assert runner.frames_plan([1280, 800], {"stack": 2, "max_side": 768}) == {"result": "skipped"}
    # D75: plain doubles, no tolerance: 1 - 0.99 is a hair above 0.01, so it is followed
    assert plan([1000, 1000], {"max_side": 2000, "crop": [0.99, 0, 0.5, 1]})["kept"] == [990, 0, 10, 1000]
    assert plan([1000, 1000], {"max_side": 2000, "crop": [0.5, 0, 0.01, 1]}) == {"result": "skipped"}
    # D76, D77: gap_ms a number of at least 0; max_side a whole number of at least 64
    assert runner.frames_plan([1280, 800], {"stack": 2, "gap_ms": -1, "max_side": 768}) == {"result": "skipped"}
    assert runner.frames_plan([1280, 800], {"stack": 2, "gap_ms": 0, "max_side": 768})["result"] == "followed"
    assert plan([1280, 800], {"max_side": 768.5}) == {"result": "skipped"}
