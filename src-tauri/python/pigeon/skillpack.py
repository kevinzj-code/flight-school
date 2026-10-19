"""Pack a staged skill pack and seal it.

A skill pack carries code, so it is never assembled here from parts this module
understands. The app that owns the code writes a STAGE — the exact files, laid
out as they will sit in the archive, with a `stage.json` recording the commit
they came from and the sha256 of every one (Pigeon's
`scripts/export-pigeon-stage.mjs`). This module only checks that the stage is
still what was recorded, zips it, and seals it (docs/PIGEON-FORMAT.md §12).

Standard library only, like the rest of this package.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import seal as seal_mod

STAGE_FILE = "stage.json"


class StageError(Exception):
    """A stage that is not what its record says, said in one sentence."""


def _staged_names(stage: Path) -> list[str]:
    names = []
    for p in sorted(stage.rglob("*")):
        if p.is_symlink():
            raise StageError(f"{p.relative_to(stage).as_posix()} is a symlink — a stage holds files, never links")
        if p.is_file():
            rel = p.relative_to(stage).as_posix()
            if rel != STAGE_FILE:
                names.append(rel)
    return names


def pack_skillpack(stage, key_path, out, *, allow_dirty: bool = False) -> dict:
    """Zip `stage` into `out`, sealed with the key at `key_path`. Returns what
    was written. Refuses a stage from uncommitted code (unless `allow_dirty`,
    for tests), a file changed, added or missing since staging, and anything
    that is not a skill pack."""
    stage, out = Path(stage), Path(out)
    record = json.loads((stage / STAGE_FILE).read_text())
    if not record.get("clean") and not allow_dirty:
        raise StageError("the stage was made from uncommitted code — commit, stage again, then pack")
    listed = record.get("files") or {}
    present = _staged_names(stage)
    extra = sorted(set(present) - set(listed))
    missing = sorted(set(listed) - set(present))
    if extra or missing:
        raise StageError(f"the stage does not match its record — added {extra[:3]}, missing {missing[:3]}")

    data = {}
    for name in present:
        blob = (stage / name).read_bytes()
        if hashlib.sha256(blob).hexdigest() != listed[name]:
            raise StageError(f"{name} changed after it was staged")
        data[name] = blob

    manifest = json.loads(data.get("pigeon.json", b"{}"))
    if manifest.get("kind") != "skillpack":
        raise StageError(f"pigeon.json is a {manifest.get('kind')!r}, not a skill pack")
    entry = (manifest.get("code") or {}).get("entry")
    if not isinstance(entry, str) or not entry.startswith("code/") or entry not in data:
        raise StageError("code.entry must name a code/ file in the stage")

    seed = seal_mod.read_key(key_path)
    order = ["pigeon.json"] + [n for n in present if n != "pigeon.json"]
    blob = seal_mod.build_sealed([(n, data[n]) for n in order], seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    part.write_bytes(blob)
    try:
        pub = seal_mod.public_key(seed)
        key = seal_mod.verify_file(part, {seal_mod.key_id(pub): pub.hex()})
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    os.replace(part, out)
    return {"path": str(out), "key": key, "entries": len(order) + 1, "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(), "head": record.get("head"),
            "clean": bool(record.get("clean"))}
