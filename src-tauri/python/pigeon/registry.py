"""The base registry — spec §7.4: where a base, its licence and its projector
are found when a Pigeon only names them.

A registry is published sealed: a strict ZIP (§3.2) holding `registry.json` and
`seal.json`, sealed exactly as a Pigeon is (§15.1). It is used only when its
seal verifies against a key the host trusts FOR REGISTRIES — a separate list
from the keys trusted for Pigeons (§15.2: one key, one purpose).

Standard library only. Nothing here downloads; it answers what to offer.
"""
from __future__ import annotations

import io
import json
import zipfile

from . import seal as seal_mod
from .gguf import same_base


REGISTRY_VERSION = 1


class RegistryRefused(Exception):
    """`code` is the spec's code: `bad_registry` for a registry that is unsealed
    or unreadable (D44), `newer_format` for a newer `registry_version`, and the
    seal's and the ZIP's own codes otherwise."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


def open_registry(path_or_bytes, registry_keys: dict | None) -> tuple[dict, str]:
    """(registry.json, the key id that sealed it). `registry_keys` is
    {key id: public key hex}, the host's list for registries."""
    data = seal_mod._bytes_of(path_or_bytes)
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise RegistryRefused("not_a_zip", "the registry is not a ZIP archive")
    try:
        entries = seal_mod.strict_entries(data)
    except seal_mod.SealError as e:
        raise RegistryRefused("bad_zip", e.reason)
    try:
        key = seal_mod.check_entries(entries, registry_keys or {})
    except seal_mod.SealError as e:
        raise RegistryRefused("untrusted_seal" if "trust" in e.reason else "broken_seal", e.reason)
    if key is None:
        raise RegistryRefused("bad_registry", "the registry is not sealed")
    try:
        reg = json.loads(dict(entries)["registry.json"])
        version = reg.get("registry_version") if isinstance(reg, dict) else None
        if isinstance(version, bool) or not isinstance(version, int) or not isinstance(reg.get("bases"), list):
            raise ValueError("it must be an object with a whole-number 'registry_version' and a 'bases' list")
    except (KeyError, ValueError) as e:
        raise RegistryRefused("bad_registry", f"registry.json could not be read — {e}")
    if version > REGISTRY_VERSION:
        raise RegistryRefused("newer_format", f"this is registry version {version}; this app reads up to "
                                              f"{REGISTRY_VERSION}. Update the app to use it")
    return reg, key


def _spellings(entry: dict) -> list[dict]:
    """The entry's identity under its own name and under each accepted spelling
    in `names` (D45)."""
    ident = entry.get("identity") or {}
    return [{**ident, "name": n} for n in [ident.get("name"), *(entry.get("names") or [])]]


def entry_for(registry: dict, identity: dict) -> dict | None:
    """The first entry that is the same base as a Pigeon's `base_model` (§7.2)."""
    for e in registry.get("bases", []):
        if isinstance(e, dict) and any(same_base(identity, c) for c in _spellings(e)):
            return e
    return None


def header_is_entry(entry: dict, header: dict) -> bool:
    """Whether an installed GGUF's identity is this entry's base."""
    return any(same_base(c, header) for c in _spellings(entry))


def disagrees(entry: dict, bases: list[dict], sha_of) -> bool:
    """D46, D61: an installed file the entry lists by sha256 whose own header
    is not the entry's base. A base whose `sha256` is already known is judged by
    it, whatever its name. Otherwise only a file carrying a listed name is
    hashed (`sha_of(base)`): a Reader is never required to hash a file just to
    look, so an import never hashes every model on the machine."""
    rows = [f for f in entry.get("files", []) if isinstance(f, dict)]
    hashes = {f.get("sha256") for f in rows if f.get("sha256")}
    by_name = {f.get("file"): f.get("sha256") for f in rows}
    for b in bases:
        if header_is_entry(entry, b):
            continue
        known = b.get("sha256")
        if known:
            if known in hashes:
                return True
        elif b.get("file") in by_name and sha_of(b) == by_name[b["file"]]:
            return True
    return False


def recommended(rows) -> dict | None:
    """The row marked `recommended` among an entry's `files` or `projectors`."""
    return next((r for r in rows or [] if isinstance(r, dict) and r.get("recommended") is True), None)


def projector_hashes(entry: dict | None) -> list[str]:
    """The stock projectors an entry lists for its base (§13.1)."""
    return [p.get("sha256") for p in (entry or {}).get("projectors", []) if isinstance(p, dict)]


def stock_projector(registry: dict | None, base_model: dict) -> bool:
    """§13.1: whether a Pigeon's projector is a stock one — its
    `mmproj.sha256` is a projector the registry lists for its base. A Writer
    leaves a stock projector out and carries every other one; with no
    registry, nothing is stock. The packer and the lineup's rewrite both ask
    this, so they never disagree about what is carried."""
    sha = (base_model.get("mmproj") or {}).get("sha256")
    if not (registry and sha):
        return False
    want = {k: base_model.get(k) for k in ("arch", "name", "size", "finetune")}
    return sha in projector_hashes(entry_for(registry, want))


def needs(want: dict, file: str, entry: dict | None, rows: str = "files") -> dict:
    """What a host needs to offer a missing base or projector and show its
    licence first (§16.4): the file and the identity always, and the registry's
    id, licence and the file's size when an entry lists the base."""
    out = {"file": file, "identity": {k: want[k] for k in ("arch", "name", "size", "finetune") if want.get(k)}}
    if entry:
        # D63: what the entry does not have is left out, never written as null
        for key, src in (("registry_id", "id"), ("license", "license"), ("license_url", "license_url")):
            if entry.get(src) is not None:
                out[key] = entry[src]
        row = next((r for r in entry.get(rows, []) if isinstance(r, dict) and r.get("file") == file), None)
        if row and row.get("bytes") is not None:
            out["bytes"] = row["bytes"]
    return out
