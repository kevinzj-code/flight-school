"""The lineup: adding a .pigeon copies it in, refused files stay out, and
"already here" is about the training, not the file name.

    python3 -m pytest src-tauri/python/bench/test_store.py -q
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import pack_run  # noqa: E402
from pigeon import store  # noqa: E402
from pigeon.validate import BASE_EXACT  # noqa: E402

from bench.test_pigeon import QWEN25, base_q4, run_folder  # noqa: E402


_N = itertools.count()


def a_pigeon(tmp_path, name="a.pigeon", kv=QWEN25, base_name="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"):
    """A packed Pigeon whose base sits in its own models folder, and whose
    adapter bytes differ from every other fixture's (the fixture GGUFs are
    header-only, so two runs would otherwise be byte-identical)."""
    models = tmp_path / f"models-{next(_N)}"; models.mkdir()
    base = base_q4(models, kv, base_name)
    rd = run_folder(tmp_path, base)
    adapter = rd / "export" / "job.lora.gguf"
    adapter.write_bytes(adapter.read_bytes() + name.encode())
    out = tmp_path / name
    pack_run(rd, out)
    return out, base


def test_add_copies_the_file_and_writes_a_sidecar(tmp_path):
    p, base = a_pigeon(tmp_path)
    st = tmp_path / "store"
    r = store.add(p, st, [base.parent])
    assert [x["status"] for x in r] == ["added"]
    pid = r[0]["id"]
    assert (st / f"{pid}.pigeon").read_bytes() == p.read_bytes()
    side = json.loads((st / f"{pid}.json").read_text())
    assert side["validation"]["base_match"] == BASE_EXACT
    assert side["source"] == "added" and side["source_path"] == str(p)
    assert p.exists(), "the original is left where it was"


def test_a_refused_file_is_not_copied(tmp_path):
    p, base = a_pigeon(tmp_path)
    st = tmp_path / "store"
    empty = tmp_path / "no-models"; empty.mkdir()
    r = store.add(p, st, [empty])
    assert r[0]["status"] == "refused" and r[0]["errors"]
    assert not list(st.glob("*")), "nothing lands in the lineup"


def test_same_training_is_already_here_whatever_the_file_is_called(tmp_path):
    p, base = a_pigeon(tmp_path)
    st = tmp_path / "store"
    assert store.add(p, st, [base.parent])[0]["status"] == "added"
    renamed = tmp_path / "copy-of-a.pigeon"; renamed.write_bytes(p.read_bytes())
    r = store.add(renamed, st, [base.parent])[0]
    assert r["status"] == "already"
    assert len(list(st.glob("*.pigeon"))) == 1


def test_same_id_different_training_keeps_both(tmp_path):
    p1, base = a_pigeon(tmp_path, "one.pigeon")
    p2, _ = a_pigeon(tmp_path, "two.pigeon")    # a fresh run folder → a fresh adapter, same job name
    st = tmp_path / "store"
    r1 = store.add(p1, st, [base.parent])[0]
    r2 = store.add(p2, st, [base.parent])[0]
    assert r1["status"] == r2["status"] == "added"
    assert r2["id"] == r1["id"] + "-2"


def test_list_reads_the_scorecard_out_of_the_zip(tmp_path):
    p, base = a_pigeon(tmp_path)
    st = tmp_path / "store"
    store.add(p, st, [base.parent])
    rows = store.list_store(st)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "coin sorter" and row["kind"] == "classifier"
    assert (row["passed"], row["total"], row["control_passed"]) == (5, 6, 1)
    assert row["base"] == base.name and row["base_match"] == BASE_EXACT
    assert row["added_ms"] and row["source"] == "added"
    assert row["engine"] is None, "an older file names no engine (D98); the card shows nothing for it"


def test_list_returns_the_card_and_a_repack_gets_the_next_revision(tmp_path):
    from pigeon.cli import _next_revision
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base)
    store_dir = tmp_path / "lineup"
    first = tmp_path / "coin-sorter.pigeon"
    pack_run(rd, first, name="coin sorter")
    store.add(first, store_dir, [tmp_path])
    row = next(r for r in store.list_store(store_dir) if r["id"] == "coin-sorter")
    assert row["card"]["does"] == "Sort coin photos by type." and row["card"]["revision"] == 1
    assert _next_revision(store_dir, "coin sorter", rd) == 2
    assert _next_revision(store_dir, "something new", rd) == 1
    assert _next_revision(None, "coin sorter", rd) == 1


def test_remove_takes_it_out_of_the_lineup(tmp_path, monkeypatch):
    p, base = a_pigeon(tmp_path)
    st = tmp_path / "store"
    trash = tmp_path / "home" / ".Trash"; trash.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    pid = store.add(p, st, [base.parent])[0]["id"]
    r = store.remove(st, pid)
    assert not (st / f"{pid}.pigeon").exists() and not (st / f"{pid}.json").exists()
    assert len(r["moved_to_trash"]) == 2 and all(Path(x).exists() for x in r["moved_to_trash"])


def test_remove_refuses_a_path_for_an_id(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        store.remove(tmp_path, "../etc")


def _pack_cli(rd, out, store_dir, models):
    """`cli pack` exactly as Send runs it (table.rs table_send), store and all."""
    import subprocess
    cli = Path(__file__).resolve().parent.parent / "pigeon" / "cli.py"
    got = subprocess.run([sys.executable, str(cli), "pack", "--run-dir", str(rd), "--out", str(out), "--name", "coin sorter",
                          "--store", str(store_dir), "--models", str(models)], capture_output=True, text=True)
    reply = json.loads(got.stdout.strip().splitlines()[-1])
    assert "error" not in reply and reply["validation"]["ok"], reply
    return reply


def _revision(p):
    import zipfile
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("pigeon.json"))["card"]["revision"]


def test_save_save_try_are_numbered_in_order(tmp_path):
    """Two files from one table never share a revision: 'Save file…' writes a
    file and leaves the lineup alone, so the table's own record of its packs
    numbers them, whichever button made each."""
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base)
    store_dir = tmp_path / "lineup"
    saved1, saved2 = tmp_path / "saved-1.pigeon", tmp_path / "saved-2.pigeon"
    _pack_cli(rd, saved1, store_dir, base.parent)                       # Save file…
    _pack_cli(rd, saved2, store_dir, base.parent)                       # Save file…
    tried = rd / "export" / "coin-sorter.pigeon"
    _pack_cli(rd, tried, store_dir, base.parent)                        # Try it in Pigeon…
    store.add(tried, store_dir, [base.parent])                          # …which adds it to the lineup
    assert [_revision(p) for p in (saved1, saved2, tried)] == [1, 2, 3]
    packs = json.loads((rd / "packs.json").read_text())["packs"]
    assert [r["revision"] for r in packs] == [1, 2, 3] and packs[0]["file"] == str(saved1)
    # Another save after the try goes on from the record, not from the lineup copy
    _pack_cli(rd, tmp_path / "saved-3.pigeon", store_dir, base.parent)
    assert _revision(tmp_path / "saved-3.pigeon") == 4


def test_a_table_without_a_record_continues_its_numbering(tmp_path):
    """The first pack after the record began continues from what the table
    produced before: the lineup copy (every pack went there), or a Pigeon in its
    own export folder, whichever is higher."""
    from pigeon.cli import _next_revision
    base = base_q4(tmp_path)
    rd = run_folder(tmp_path, base)
    store_dir = tmp_path / "lineup"
    old = tmp_path / "old.pigeon"
    pack_run(rd, old, name="coin sorter", revision=4)                   # made before the record existed
    store.add(old, store_dir, [base.parent])
    assert _next_revision(store_dir, "coin sorter", rd) == 5
    pack_run(rd, rd / "export" / "coin-sorter.pigeon", name="coin sorter", revision=6)
    assert _next_revision(store_dir, "coin sorter", rd) == 7            # the export folder's is higher
    assert _next_revision(None, "coin sorter", rd) == 7                 # and it needs no lineup
    _pack_cli(rd, tmp_path / "next.pigeon", store_dir, base.parent)
    assert _revision(tmp_path / "next.pigeon") == 7
    assert json.loads((rd / "packs.json").read_text())["packs"][-1]["revision"] == 7
