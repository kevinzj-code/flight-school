"""The .nest: a flock in one file, and adding one adds every good Pigeon
inside — the bad one named, not fatal.

    python3 -m pytest src-tauri/python/bench/test_nest.py -q
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import nest, store  # noqa: E402

from bench.test_store import a_pigeon  # noqa: E402
from bench.test_pigeon import QWEN35  # noqa: E402


def test_pack_puts_pigeons_at_the_root_in_store_mode_with_a_listing(tmp_path):
    p1, base = a_pigeon(tmp_path, "one.pigeon")
    p2, _ = a_pigeon(tmp_path, "two.pigeon")
    out = tmp_path / "flock.nest"
    m = nest.pack(out, [p1, p2], "my flock")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert set(names) == {"nest.json", "coin-sorter.pigeon", "coin-sorter-2.pigeon"}
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
        assert z.read("coin-sorter.pigeon") == p1.read_bytes()
    assert m["format_version"] == 1 and m["name"] == "my flock"
    # members are named by manifest id, never by the file name they arrived with
    assert [e["file"] for e in m["pigeons"]] == ["coin-sorter.pigeon", "coin-sorter-2.pigeon"]
    assert m["pigeons"][0]["name"] == "coin sorter" and m["pigeons"][0]["base_file"] == base.name


def test_list_and_unpack_round_trip(tmp_path):
    p1, _ = a_pigeon(tmp_path, "one.pigeon")
    out = tmp_path / "flock.nest"
    nest.pack(out, [p1])
    listed = nest.list_nest(out)
    assert listed["members"] == ["coin-sorter.pigeon"]
    got = nest.unpack(out, tmp_path / "unpacked")
    assert [g.name for g in got] == ["coin-sorter.pigeon"] and got[0].read_bytes() == p1.read_bytes()


def test_a_zip_that_is_not_a_nest_says_so(tmp_path):
    p1, _ = a_pigeon(tmp_path, "one.pigeon")
    byhand = tmp_path / "byhand.nest"
    with zipfile.ZipFile(byhand, "w") as z:
        z.write(p1, "one.pigeon")
    with pytest.raises(nest.NotANest, match="nest.json"):
        nest.list_nest(byhand)
    (tmp_path / "text.nest").write_text("hello")
    with pytest.raises(nest.NotANest, match="ZIP"):
        nest.list_nest(tmp_path / "text.nest")


def test_a_newer_nest_format_is_refused(tmp_path):
    p1, _ = a_pigeon(tmp_path, "one.pigeon")
    out = tmp_path / "flock.nest"
    nest.pack(out, [p1])
    newer = tmp_path / "newer.nest"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(newer, "w") as dst:
        for n in src.namelist():
            data = src.read(n)
            if n == "nest.json":
                d = json.loads(data); d["format_version"] = 99; data = json.dumps(d).encode()
            dst.writestr(n, data)
    with pytest.raises(nest.NotANest, match="99"):
        nest.list_nest(newer)


def test_adding_a_nest_adds_the_good_and_names_the_bad(tmp_path):
    p1, base = a_pigeon(tmp_path, "one.pigeon")
    # a Pigeon for a model this Mac does not have
    p2, _ = a_pigeon(tmp_path, "two.pigeon", kv=QWEN35, base_name="Qwen3.5-4B-Q4_K_M.gguf")
    out = tmp_path / "flock.nest"
    nest.pack(out, [p1, p2])
    st = tmp_path / "store"
    r = store.add(out, st, [base.parent])
    by = {Path(x["file"]).name: x for x in r}
    assert by["coin-sorter.pigeon"]["status"] == "added" and by["coin-sorter.pigeon"]["from_nest"] == "flock.nest"
    assert by["coin-sorter-2.pigeon"]["status"] == "refused" and "Qwen3.5" in " ".join(by["coin-sorter-2.pigeon"]["errors"])
    assert len(list(st.glob("*.pigeon"))) == 1
    assert not list(Path(tmp_path).glob("nest-*")), "temporaries are cleaned up"


def test_pack_needs_real_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        nest.pack(tmp_path / "x.nest", [tmp_path / "missing.pigeon"])
    with pytest.raises(ValueError):
        nest.pack(tmp_path / "x.nest", [])
