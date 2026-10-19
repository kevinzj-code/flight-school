"""pack-skillpack: a stage is packed only while it is exactly what its record says."""

import hashlib
import json
import os
import zipfile

import pytest

from pigeon import seal as seal_mod
from pigeon.skillpack import StageError, pack_skillpack


def _stage(tmp_path, *, clean=True, kind="skillpack"):
    stage = tmp_path / "stage"
    files = {
        "pigeon.json": json.dumps({"format_version": 1, "name": "Patch", "id": "patch", "kind": kind,
                                   "version": 1, "models": {}, "code": {"entry": "code/skills/index.js"}}).encode(),
        "code/skills/index.js": b"export default [];\n",
        "code/src/lib/num.js": b"export const one = 1;\n",
        "prompts/planner.json": b'{"system":"hi"}\n',
        "adapters/category.lora.gguf": b"GGUF not really",
    }
    for name, data in files.items():
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        (stage / name).write_bytes(data)
    record = {"head": "abc123", "clean": clean, "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}}
    (stage / "stage.json").write_text(json.dumps(record))
    key = tmp_path / "keys" / "seal.key"
    seal_mod.keygen(key)
    return stage, key


def test_a_stage_packs_manifest_first_seal_last_and_verifies(tmp_path):
    stage, key = _stage(tmp_path)
    out = tmp_path / "patch.pigeon"
    r = pack_skillpack(stage, key, out)
    names = [n for n, _ in seal_mod.strict_entries(out)]
    assert names[0] == "pigeon.json" and names[-1] == "seal.json"
    assert "stage.json" not in names and len(names) == r["entries"] == 6
    pub = seal_mod.public_key(seal_mod.read_key(key))
    assert seal_mod.verify_file(out, {seal_mod.key_id(pub): pub.hex()}) == r["key"]
    assert r["head"] == "abc123" and not out.with_name("patch.pigeon.part").exists()
    with zipfile.ZipFile(out) as z:
        assert z.read("code/src/lib/num.js") == b"export const one = 1;\n"


def test_a_file_changed_after_staging_is_refused(tmp_path):
    stage, key = _stage(tmp_path)
    (stage / "code/src/lib/num.js").write_bytes(b"export const one = 2;\n")
    with pytest.raises(StageError, match="changed after it was staged"):
        pack_skillpack(stage, key, tmp_path / "p.pigeon")


def test_an_added_or_missing_file_is_refused(tmp_path):
    stage, key = _stage(tmp_path)
    (stage / "code/extra.js").write_bytes(b"x")
    with pytest.raises(StageError, match="does not match its record"):
        pack_skillpack(stage, key, tmp_path / "p.pigeon")
    os.remove(stage / "code/extra.js")
    os.remove(stage / "prompts/planner.json")
    with pytest.raises(StageError, match="does not match its record"):
        pack_skillpack(stage, key, tmp_path / "p.pigeon")


def test_a_stage_from_uncommitted_code_is_refused_unless_a_test_says_so(tmp_path):
    stage, key = _stage(tmp_path, clean=False)
    with pytest.raises(StageError, match="uncommitted"):
        pack_skillpack(stage, key, tmp_path / "p.pigeon")
    assert pack_skillpack(stage, key, tmp_path / "p.pigeon", allow_dirty=True)["clean"] is False


def test_only_a_skill_pack_is_packed_this_way(tmp_path):
    stage, key = _stage(tmp_path, kind="freeform")
    with pytest.raises(StageError, match="not a skill pack"):
        pack_skillpack(stage, key, tmp_path / "p.pigeon")


def test_a_symlink_in_the_stage_is_refused(tmp_path):
    stage, key = _stage(tmp_path)
    (stage / "code/link.js").symlink_to(stage / "code/skills/index.js")
    with pytest.raises(StageError, match="symlink"):
        pack_skillpack(stage, key, tmp_path / "p.pigeon")
