"""The sealed registry inside the app (D73) and small picture Pigeons (§13.1).

A picture Pigeon names its projector by hash either way. It carries the
projector only when the app's registry does not list it for the base — a
projector the registry lists is installed with the base, or downloaded from
the registry. With no registry, or one that does not open, every projector is
carried (D44: a bad registry is ignored, never fatal).

    python3 -m pytest src-tauri/python/bench/test_sealed_registry.py -q
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from bench import make_seal_fixture as fixture  # noqa: E402
from bench.test_pigeon import QWEN35, base_q4, run_folder, sha, write_gguf  # noqa: E402
from pigeon import base_identity, host, pack_run, read_header, registry, seal, store, validate_pigeon  # noqa: E402
from pigeon.hashes import HashCache  # noqa: E402
from pigeon.pack import drop_stock_projector  # noqa: E402
from pigeon.validate import installed_bases  # noqa: E402

APP = Path(__file__).resolve().parents[2]
TEST_KEY = seal.key_id(fixture.TEST_PUB)


def key_list(tmp_path, rows) -> Path:
    p = tmp_path / "registry-keys.json"
    p.write_text(json.dumps(rows))
    return p


def key_row(**over) -> dict:
    return {"id": TEST_KEY, "ed25519_pub_hex": fixture.TEST_PUB.hex(), "revoked": False, **over}


def sealed_registry(tmp_path, base: Path, projector_sha: str) -> Path:
    ident = base_identity(read_header(base))
    reg = {"registry_version": 1, "bases": [{
        "id": "qwen3.5-4b", "identity": {k: ident.get(k) for k in ("arch", "name", "size", "finetune")},
        "license": "Apache-2.0",
        "files": [{"file": base.name, "sha256": sha(base), "recommended": True}],
        "projectors": [{"file": "mmproj.gguf", "sha256": projector_sha, "bytes": 123, "recommended": True}]}]}
    out = tmp_path / "registry.zip"
    out.write_bytes(seal.build_sealed([("registry.json", json.dumps(reg).encode())], fixture.TEST_SEED))
    return out


def vision_run(tmp_path):
    """A picture run on Qwen3.5-4B with its projector installed with the base,
    in `projectors/` as Pigeon's store keeps it."""
    models = tmp_path / "models"; models.mkdir()
    base = base_q4(models, QWEN35, "Qwen3.5-4B-Q4_K_M.gguf")
    (models / "projectors").mkdir()
    mm = write_gguf(models / "projectors" / "Qwen3.5-4B-mmproj-F16.gguf", {"general.architecture": "clip"}, kind="mmproj")
    rd = run_folder(tmp_path, base, vision=True)
    (rd / "config.toml").write_text(f'[models]\ntarget_mmproj = "{mm}"\n')
    return models, base, mm, rd


def opened(tmp_path, base, projector_sha):
    zp = sealed_registry(tmp_path, base, projector_sha)
    return host.open_bundled(zp, key_list(tmp_path, [key_row()]))


# ---------------------------------------------------------------------------
# the packer


def test_a_stock_projector_is_named_but_not_carried_and_the_gate_still_accepts(tmp_path):
    models, base, mm, rd = vision_run(tmp_path)
    reg = opened(tmp_path, base, sha(mm))
    out = tmp_path / "v.pigeon"
    m = pack_run(rd, out, registry=reg)
    assert m["base_model"]["mmproj"] == {"file": mm.name, "sha256": sha(mm)}, "the manifest names it either way"
    with zipfile.ZipFile(out) as z:
        assert "projector.gguf" not in z.namelist()
    # The rebuilt-file check, as a fresh install runs it: with the registry,
    # finding the projector installed with the base.
    installed = host.installed_projectors([models], tmp_path / "hashes.json")
    assert installed == [sha(mm)]
    r = validate_pigeon(out, installed_bases([models]), projectors=installed, registry=reg)
    assert r.ok, r.errors
    # With no projector installed, the registry still offers it: accepted.
    r = validate_pigeon(out, installed_bases([models]), projectors=[], registry=reg)
    assert r.ok, r.errors
    # With no projector and no registry, there is nowhere to get it.
    r = validate_pigeon(out, installed_bases([models]), projectors=[], registry=None)
    assert not r.ok and r.code == "no_projector"


def test_a_projector_the_registry_does_not_list_is_carried(tmp_path):
    models, base, mm, rd = vision_run(tmp_path)
    reg = opened(tmp_path, base, "e" * 64)          # lists a different projector for this base
    out = tmp_path / "v.pigeon"
    pack_run(rd, out, registry=reg)
    with zipfile.ZipFile(out) as z:
        assert z.read("projector.gguf") == mm.read_bytes()
    r = validate_pigeon(out, installed_bases([models]), projectors=[], registry=reg)
    assert r.ok, r.errors


def test_with_no_registry_or_one_that_does_not_open_every_projector_is_carried(tmp_path, capsys):
    models, base, mm, rd = vision_run(tmp_path)
    zp = sealed_registry(tmp_path, base, sha(mm))
    # Sealed by a key on no list: refused, logged, not used.
    assert host.open_bundled(zp, key_list(tmp_path, [])) is None
    assert "[registry]" in capsys.readouterr().err
    # Not a registry at all.
    junk = tmp_path / "junk.zip"; junk.write_bytes(b"not a zip")
    assert host.open_bundled(junk, key_list(tmp_path, [key_row()])) is None
    # Missing.
    assert host.open_bundled(tmp_path / "gone.zip", key_list(tmp_path, [key_row()])) is None
    for reg in (None, host.open_bundled(junk, None)):
        out = tmp_path / "v.pigeon"
        pack_run(rd, out, registry=reg)
        with zipfile.ZipFile(out) as z:
            assert "projector.gguf" in z.namelist()


def test_a_text_pigeon_on_a_vision_base_is_unaffected(tmp_path):
    models, base, mm, _ = vision_run(tmp_path)
    reg = opened(tmp_path, base, sha(mm))
    out = tmp_path / "t.pigeon"
    pack_run(run_folder(tmp_path, base), out, registry=reg)
    with zipfile.ZipFile(out) as z:
        assert "projector.gguf" not in z.namelist()
        assert "mmproj" not in json.loads(z.read("pigeon.json"))["base_model"]


# ---------------------------------------------------------------------------
# the lineup's rewrite


def test_the_rewrite_drops_only_the_stock_projector_and_copies_the_rest_byte_for_byte(tmp_path):
    models, base, mm, rd = vision_run(tmp_path)
    fat = tmp_path / "fat.pigeon"
    pack_run(rd, fat)                                # packed before the registry: carries it
    reg = opened(tmp_path, base, sha(mm))
    slim = tmp_path / "slim.pigeon"
    assert drop_stock_projector(fat, slim, reg) == {"dropped": True}
    with zipfile.ZipFile(fat) as a, zipfile.ZipFile(slim) as b:
        before = [i for i in a.infolist() if i.filename != "projector.gguf"]
        after = b.infolist()
        assert [i.filename for i in before] == [i.filename for i in after], "same entries, same order"
        for x, y in zip(before, after):
            assert a.read(x) == b.read(y) and x.date_time == y.date_time and x.compress_type == y.compress_type
    r = validate_pigeon(slim, installed_bases([models]), projectors=[sha(mm)], registry=reg)
    assert r.ok, r.errors


def test_the_rewrite_leaves_alone_what_it_must_carry(tmp_path):
    models, base, mm, rd = vision_run(tmp_path)
    fat = tmp_path / "fat.pigeon"
    pack_run(rd, fat)
    out = tmp_path / "out.pigeon"
    assert not drop_stock_projector(fat, out, opened(tmp_path, base, "e" * 64))["dropped"]
    assert not drop_stock_projector(fat, out, None)["dropped"]
    assert not out.exists(), "nothing is written when nothing is dropped"
    # A sealed file's seal covers its projector: refused, never broken.
    key = tmp_path / "test.key"; key.write_text(fixture.TEST_SEED.hex() + "\n"); key.chmod(0o600)
    seal.seal_file(fat, key)
    with pytest.raises(ValueError, match="sealed"):
        drop_stock_projector(fat, out, opened(tmp_path, base, sha(mm)))


def test_d74_the_lineup_keeps_the_smaller_copy_of_the_same_pigeon(tmp_path):
    models, base, mm, rd = vision_run(tmp_path)
    reg = opened(tmp_path, base, sha(mm))
    fat, slim = tmp_path / "fat.pigeon", tmp_path / "slim.pigeon"
    pack_run(rd, fat)
    drop_stock_projector(fat, slim, reg)
    lineup = tmp_path / "lineup"
    kw = {"registry": reg, "projectors": [sha(mm)]}
    [first] = store.add(fat, lineup, [models], **kw)
    assert first["status"] == "added"
    held = lineup / f"{first['id']}.pigeon"
    [again] = store.add(slim, lineup, [models], **kw)
    assert again["status"] == "already" and again["kept"] == "smaller"
    assert held.read_bytes() == slim.read_bytes()
    [back] = store.add(fat, lineup, [models], **kw)
    assert back["status"] == "already" and "kept" not in back
    assert held.read_bytes() == slim.read_bytes(), "the larger copy never replaces the smaller"


# ---------------------------------------------------------------------------
# what the app trusts and has


def test_a_registry_key_whose_written_id_disagrees_is_ignored_and_logged(tmp_path, capsys):
    rows = [key_row(id="0000000000000000"), key_row(revoked=True)]
    assert host.registry_keys(key_list(tmp_path, rows)) == {}
    assert "is 21fe31dfa154a261 by its key" in capsys.readouterr().err
    assert host.registry_keys(key_list(tmp_path, [key_row()])) == {TEST_KEY: fixture.TEST_PUB.hex()}
    assert host.registry_keys(None) == {}


def test_the_shipped_registry_opens_under_the_shipped_key_and_only_that_key(tmp_path):
    keys = host.registry_keys(APP / "keys" / "registry-keys.json")
    assert list(keys) == ["25660ed562d33a0d"], "Kevin's registry-only key, and nothing else"
    assert TEST_KEY not in keys, "the test key is trusted only in tests"
    reg = host.open_bundled(APP / "registry" / "registry.zip", APP / "keys" / "registry-keys.json")
    assert reg is not None
    vl2b = {"arch": "qwen3vl", "name": "qwen3vl", "size": "2B", "finetune": "instruct",
            "mmproj": {"sha256": "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"}}
    assert registry.stock_projector(reg, vl2b)
    assert not registry.stock_projector(reg, {**vl2b, "mmproj": {"sha256": "0" * 64}})


def test_projector_hashes_are_cached_by_the_app_never_beside_the_models(tmp_path):
    models, base, mm, _ = vision_run(tmp_path)
    cache = tmp_path / "app" / "pigeon-hashes.json"
    before = sorted(p.name for p in (models / "projectors").iterdir())
    assert host.installed_projectors([models], cache) == [sha(mm)]
    assert sorted(p.name for p in (models / "projectors").iterdir()) == before
    assert HashCache(cache).known(mm) == sha(mm)
    mm.write_bytes(mm.read_bytes() + b"x")           # changed: not believed
    assert HashCache(cache).known(mm) is None
    assert host.installed_projectors([models], cache) == [sha(mm)]


# ---------------------------------------------------------------------------
# the Send page's "What goes in the file" (pack.file_parts)


def test_the_preview_total_is_the_packed_size_and_follows_the_packers_rule(tmp_path):
    from pigeon.pack import file_parts
    models, base, mm, rd = vision_run(tmp_path)
    for listed, carried in ((sha(mm), False), ("e" * 64, True)):
        reg = opened(tmp_path, base, listed)
        parts = file_parts(rd, registry=reg, hash_cache=tmp_path / "hashes.json")
        out = tmp_path / f"v-{carried}.pigeon"
        pack_run(rd, out, registry=reg)
        with zipfile.ZipFile(out) as z:
            assert ("projector.gguf" in z.namelist()) is carried
        assert parts["projector_carried"] is carried
        assert parts["projector"] == mm.stat().st_size, "the projector's size is known either way"
        # Within 50 KB: the card's own files (pigeon.json, icon, README, report) are estimated.
        assert abs(parts["total"] - out.stat().st_size) < 50_000, (parts, out.stat().st_size)


def test_the_preview_names_the_base_the_run_trained_on():
    from pigeon.pack import base_display_name
    assert base_display_name("Qwen2-VL-2B-Instruct-Q4_K_M.gguf") == "Qwen2-VL-2B"
    assert base_display_name("Qwen3VL-2B-Instruct-Q4_K_M.gguf") == "Qwen3-VL-2B"
    assert base_display_name("Qwen3.5-4B-Q4_K_M.gguf") == "Qwen3.5-4B"
    assert base_display_name("Qwen2.5-1.5B-Instruct-F16.gguf") == "Qwen2.5-1.5B"


def test_a_tables_preview_names_its_own_base_not_a_default(tmp_path):
    from pigeon.pack import file_parts
    models, base, mm, rd = vision_run(tmp_path)
    parts = file_parts(rd, registry=opened(tmp_path, base, sha(mm)), hash_cache=tmp_path / "h.json")
    assert parts["base"] == {"name": "Qwen3.5-4B", "file": base.name, "bytes": base.stat().st_size}
