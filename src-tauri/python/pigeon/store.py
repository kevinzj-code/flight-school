"""The lineup — the app's own folder of Pigeons.

A `.pigeon` that arrives by AirDrop or on a flash drive has to be *added*:
validated, then COPIED into `pigeons/` so the original can move and the drive
can be unplugged. A run that packages drops a copy here too. The library
reads this folder and nothing else, so "what Pigeons do I have" has one
answer.

    pigeons/<id>.pigeon     the file, unchanged
    pigeons/<id>.json       when and where it came from, and what the
                            importer said about it

Refused files are never copied — a file the apps would refuse has no place
in the lineup, and the person is told why, naming the model needed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

from . import nest as nest_mod
from .validate import installed_bases, validate_pigeon


def _adapter_sha(path: Path) -> str | None:
    """Fingerprint of the training itself, so "already added" is about the
    adapter, not the file name."""
    try:
        with zipfile.ZipFile(path) as z:
            return hashlib.sha256(z.read("adapter.lora.gguf")).hexdigest()
    except (KeyError, zipfile.BadZipFile, OSError):
        return None


def _read_member(path: Path, name: str):
    try:
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read(name))
    except (KeyError, zipfile.BadZipFile, OSError, ValueError):
        return None


# An icon bigger than this is left off the listing (the card falls back to
# initials); a drawn look is a few KB, a chosen PNG rarely more than a few hundred.
ICON_LIST_MAX = 512 * 1024


def icon_data_url(png: bytes | None) -> str | None:
    """A PNG as the data URL a card draws, or None when it is not a PNG or too
    big to send across the bridge on every listing. Pigeon accepts exactly this
    shape (its avatarHtml checks `data:image/png;base64,`)."""
    if not png or len(png) > ICON_LIST_MAX or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _icon(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as z:
            return icon_data_url(z.read("icon.png"))
    except (KeyError, zipfile.BadZipFile, OSError):
        return None


def _sidecar(store: Path, pid: str) -> Path:
    return store / f"{pid}.json"


def _same_but_projector(a: Path, b: Path) -> bool:
    """D74: two files that differ only in whether they carry a projector —
    every other entry the same name with the same CRC."""
    def entries(p):
        with zipfile.ZipFile(p) as z:
            return {i.filename: i.CRC for i in z.infolist() if i.filename != "projector.gguf"}
    try:
        return entries(a) == entries(b)
    except (zipfile.BadZipFile, OSError):
        return False


def _existing(store: Path) -> dict[str, dict]:
    """id -> {path, adapter_sha} for every Pigeon already in the lineup."""
    out = {}
    for p in sorted(Path(store).glob("*.pigeon")):
        out[p.stem] = {"path": p, "adapter_sha": _adapter_sha(p)}
    return out


def add_pigeon(path: Path | str, store: Path | str, models=None, *,
               source: str = "added", run_id: str | None = None,
               bases: list[dict] | None = None, registry: dict | None = None,
               projectors: list[str] | None = None) -> dict:
    """Validate one `.pigeon` and, if the apps would accept it, copy it in.

    Returns {status: added | already | refused, ...}. `bases` may be passed
    to avoid re-reading every installed model per file. `registry` and
    `projectors` are the import check's (validate_pigeon): the app's opened
    registry and the projectors installed with the bases.
    """
    path, store = Path(path), Path(store)
    store.mkdir(parents=True, exist_ok=True)
    if bases is None:
        bases = installed_bases(models or [])
    report = validate_pigeon(path, bases, projectors=projectors, registry=registry)
    name = (report.manifest or {}).get("name") or path.stem
    if not report.ok:
        return {"status": "refused", "file": str(path), "name": name,
                "base_needed": report.base_needed, "errors": report.errors, "warnings": report.warnings}

    pid = str(report.manifest.get("id") or path.stem)
    sha = _adapter_sha(path)
    have = _existing(store)
    # Same training already here, under this id or another: nothing to do —
    # except D74: the same Pigeon without its stock projector replaces the
    # copy that carries it, since this app runs both and keeps the smaller.
    for other_id, info in have.items():
        if sha and info["adapter_sha"] == sha:
            out = {"status": "already", "id": other_id, "name": name, "file": str(path),
                   "base_match": report.base_match, "warnings": report.warnings}
            if _same_but_projector(path, info["path"]) and path.stat().st_size < info["path"].stat().st_size:
                tmp = info["path"].with_suffix(".pigeon.part")
                shutil.copy2(path, tmp)
                tmp.replace(info["path"])
                out["kept"] = "smaller"
            return out
    # Same id, different training: keep both, number the newcomer.
    base_id, n = pid, 2
    while pid in have:
        pid = f"{base_id}-{n}"; n += 1

    dest = store / f"{pid}.pigeon"
    tmp = dest.with_suffix(".pigeon.part")
    shutil.copy2(path, tmp)
    tmp.rename(dest)
    _sidecar(store, pid).write_text(json.dumps({
        "id": pid, "added_ms": int(time.time() * 1000), "source": source,
        "source_path": str(path), "run_id": run_id,
        "validation": {"base_match": report.base_match, "warnings": report.warnings,
                       "self_test": report.self_test, "examples": report.examples},
    }, indent=2) + "\n")
    return {"status": "added", "id": pid, "name": name, "file": str(path),
            "base_match": report.base_match, "warnings": report.warnings}


def _declared_kind(path: Path) -> str | None:
    """The kind a file's manifest declares, judging nothing else."""
    try:
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("pigeon.json")).get("kind")
    except (KeyError, zipfile.BadZipFile, OSError, ValueError, AttributeError):
        return None


def add(path: Path | str, store: Path | str, models=None, *, source: str = "added",
        run_id: str | None = None, registry: dict | None = None,
        projectors: list[str] | None = None) -> list[dict]:
    """Add a `.pigeon`, or every Pigeon inside a `.nest`. One result per file."""
    path = Path(path)
    bases = installed_bases(models or [])
    if path.suffix.lower() == ".nest" or (path.suffix.lower() != ".pigeon" and nest_mod.is_nest(path)):
        try:
            members = nest_mod.unpack(path)
        except nest_mod.NotANest as e:
            return [{"status": "refused", "file": str(path), "name": path.name, "errors": [str(e)], "warnings": []}]
        results = []
        for m in members:
            if _declared_kind(m) == "skillpack":
                # Refused as a member, not as the nest: the good ones still land.
                results.append({"status": "refused", "file": str(m), "name": m.name,
                                "errors": [nest_mod.SKILLPACK_IN_NEST], "warnings": [],
                                "from_nest": path.name})
                continue
            r = add_pigeon(m, store, bases=bases, source=f"{source}:nest", run_id=run_id,
                           registry=registry, projectors=projectors)
            r["from_nest"] = path.name
            results.append(r)
        nest_mod.cleanup(members)
        return results
    return [add_pigeon(path, store, bases=bases, source=source, run_id=run_id,
                       registry=registry, projectors=projectors)]


def list_store(store: Path | str) -> list[dict]:
    """Every Pigeon in the lineup, with what a card needs: the manifest's
    name/kind/base, the scorecard out of report.json if there is one, and the
    sidecar's provenance. Read straight out of the zip — store mode, so this
    is a directory listing's worth of work."""
    store = Path(store)
    out = []
    for p in sorted(store.glob("*.pigeon")):
        m = _read_member(p, "pigeon.json") or {}
        rep = _read_member(p, "report.json") or {}
        ride = (rep.get("check_ride") or {}) if isinstance(rep, dict) else {}
        side = {}
        sc = _sidecar(store, p.stem)
        if sc.exists():
            try:
                side = json.loads(sc.read_text())
            except ValueError:
                side = {}
        bm = m.get("base_model") or {}
        out.append({
            "id": p.stem, "path": str(p), "bytes": p.stat().st_size,
            "name": m.get("name") or p.stem, "kind": m.get("kind"),
            # The icon the file carries, drawn as Pigeon draws it; and the
            # skill it answers with, so the Send screen knows an Icon Maker is here.
            "icon": _icon(p), "skill": (m.get("skill") or {}).get("name") if isinstance(m.get("skill"), dict) else None,
            "base": bm.get("file"), "vision": bool(bm.get("vision")),
            "trained_on": m.get("trained_on"), "trained_with": m.get("trained_with"),
            "example_count": m.get("example_count"),
            # The maker's card, when the file carries one (PIGEON-FORMAT §2.1).
            "card": m.get("card") if isinstance(m.get("card"), dict) else None,
            "passed": ride.get("passed"), "total": ride.get("total"),
            "control_passed": ride.get("control_passed"),
            "graded_by_model": ride.get("graded_by_model"),
            # D98: the engine its score was asked through, when it was.
            "engine": ride.get("engine"),
            "has_report": bool(ride),
            "added_ms": side.get("added_ms"), "source": side.get("source"),
            "run_id": side.get("run_id"),
            "base_match": (side.get("validation") or {}).get("base_match"),
            "warnings": (side.get("validation") or {}).get("warnings") or [],
        })
    out.sort(key=lambda r: -(r["added_ms"] or 0))
    return out


def find_by_skill(store: Path | str, skill: str) -> dict | None:
    """The newest Pigeon in the lineup whose manifest names `skill`, or None.

    How Flight School finds the Icon Maker (skill `pigeon-look`) without a
    setting: add one to the lineup and the package step starts asking it.
    Newest by when it was added, so a retrained one replaces the last."""
    best = None
    for row in list_store(store):
        m = _read_member(Path(row["path"]), "pigeon.json") or {}
        if (m.get("skill") or {}).get("name") == skill:
            added = row["added_ms"] or int(Path(row["path"]).stat().st_mtime * 1000)
            if best is None or added > best["added_ms"]:
                best = {"id": row["id"], "path": row["path"], "name": row["name"], "added_ms": added}
    return best


def remove(store: Path | str, pid: str) -> dict:
    """Out of the lineup and into the Trash — never deleted outright."""
    store = Path(store)
    if "/" in pid or "\\" in pid or ".." in pid:
        raise ValueError("bad id")
    src = store / f"{pid}.pigeon"
    if not src.exists():
        raise FileNotFoundError(f"{pid} is not in the lineup")
    trash = Path.home() / ".Trash"
    trash.mkdir(exist_ok=True)
    moved = []
    for f in (src, _sidecar(store, pid)):
        if f.exists():
            dest = trash / f.name
            n = 2
            while dest.exists():
                dest = trash / f"{f.stem}-{n}{f.suffix}"; n += 1
            shutil.move(str(f), str(dest))
            moved.append(str(dest))
    return {"id": pid, "moved_to_trash": moved}
