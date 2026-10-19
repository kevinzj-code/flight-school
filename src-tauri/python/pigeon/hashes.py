"""What a model file's sha256 is, without hashing gigabytes every time —
the same cache Pigeon keeps (pigeon-core `hashes.rs`, INTERFACE.md §5.1).

One file per host, at a path the host names, mapping a model file's path to
its size, its modified time and its sha256. Nothing is ever written into a
model folder. An entry is believed only while the file's size and modified
time are still the ones it was hashed at; otherwise it is hashed again.

A cache that is missing or unreadable is an empty one. It is never a reason
for anything to fail; at worst a file is hashed again.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .gguf import sha256_file

VERSION = 1


def _stamp(file: Path) -> tuple[int, int]:
    st = os.stat(file)
    return st.st_size, st.st_mtime_ns


def _key(file: Path) -> str:
    try:
        return str(Path(file).resolve())
    except OSError:
        return str(file)


class HashCache:
    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self.files: dict[str, dict] = {}
        self.dirty = False
        if self.path:
            try:
                d = json.loads(self.path.read_text())
                if d.get("version") == VERSION and isinstance(d.get("files"), dict):
                    self.files = d["files"]
            except (OSError, ValueError, AttributeError):
                pass

    def known(self, file: Path) -> str | None:
        e = self.files.get(_key(file))
        try:
            size, mtime = _stamp(file)
        except OSError:
            return None
        if isinstance(e, dict) and e.get("bytes") == size and e.get("modified_ns") == mtime:
            return e.get("sha256")
        return None

    def hash(self, file: Path) -> str:
        sha = self.known(file)
        if sha:
            return sha
        before = _stamp(file)
        sha = sha256_file(file)
        if _stamp(file) == before:     # a file that changed while it was read is not believed
            entry = {"bytes": before[0], "modified_ns": before[1], "sha256": sha}
            if self.files.get(_key(file)) != entry:
                self.files[_key(file)] = entry
                self.dirty = True
        return sha

    def save(self) -> None:
        if not (self.dirty and self.path):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".part")
            tmp.write_text(json.dumps({"version": VERSION, "files": self.files}, indent=2, sort_keys=True))
            tmp.replace(self.path)
            self.dirty = False
        except OSError:
            pass   # costs a re-hash later, nothing more
