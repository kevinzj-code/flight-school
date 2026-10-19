"""What this app trusts and has, as the import check and the packer need it:
the key list for REGISTRIES, the sealed base registry the app ships with
(D73), and the projectors installed with the bases.

Rust names the files (cli.py's top-level --registry, --registry-keys and
--hash-cache); this module only opens them. Anything wrong here is logged to
stderr with a `[registry]` prefix and never fatal (D44): with no registry the
packer carries every projector, and the import check works without it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import registry as registry_mod
from . import seal as seal_mod
from .hashes import HashCache


def log(line: str) -> None:
    print(f"[registry] {line}", file=sys.stderr)


def registry_keys(path) -> dict:
    """{key id: public key hex} from a key list in Pigeon's shape:
    `[{"id", "ed25519_pub_hex", "revoked"}]`. The id is derived from the key
    (§15.2); an entry whose written id disagrees is ignored, and so is a
    revoked one. A list that cannot be read trusts nothing."""
    if not path:
        return {}
    try:
        rows = json.loads(Path(path).read_text())
        if not isinstance(rows, list):
            raise ValueError("it must be a list")
    except (OSError, ValueError) as e:
        log(f"the registry key list {path} could not be read, so no registry is trusted — {e}")
        return {}
    out = {}
    for row in rows:
        pub = row.get("ed25519_pub_hex") if isinstance(row, dict) else None
        try:
            raw = bytes.fromhex(pub)
            if len(raw) != 32:
                raise ValueError
        except (TypeError, ValueError):
            log(f"a registry key entry is not a 32-byte public key, and is ignored: {row!r}")
            continue
        kid = seal_mod.key_id(raw)
        if row.get("id") != kid:
            log(f"the registry key written as {row.get('id')!r} is {kid} by its key; the entry is ignored")
            continue
        if row.get("revoked") is True:
            continue
        out[kid] = raw.hex()
    return out


def open_bundled(registry_path, keys_path) -> dict | None:
    """The app's sealed registry, opened under the registry key list, or None
    — with why in the log — when it cannot be used (D44)."""
    if not registry_path:
        return None
    keys = registry_keys(keys_path)
    try:
        reg, _ = registry_mod.open_registry(Path(registry_path), keys)
    except registry_mod.RegistryRefused as e:
        log(f"the registry {registry_path} is not used ({e.code}): {e.reason}")
        return None
    except OSError as e:
        log(f"the registry {registry_path} could not be read, and is not used — {e}")
        return None
    return reg


def projector_files(model_dirs) -> list[Path]:
    """Every projector installed with the bases: `projectors/*.gguf` beside
    the models, where Pigeon's store keeps them (pigeonstore.rs)."""
    return [p for d in model_dirs or [] for p in sorted((Path(d) / "projectors").glob("*.gguf"))]


def installed_projectors(model_dirs, cache_path) -> list[str]:
    """The sha256 of each installed projector. Hashes are kept in the app's
    own cache file, never beside the models."""
    cache = HashCache(cache_path)
    out = []
    for p in projector_files(model_dirs):
        try:
            out.append(cache.hash(p))
        except OSError:
            continue
    cache.save()
    return out
