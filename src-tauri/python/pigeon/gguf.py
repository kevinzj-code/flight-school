"""Read a GGUF file's header — just enough to say what model it is.

The base-match rule (spec §3.7) needs a model's IDENTITY separately from its
FILE: the same Qwen2.5-1.5B at Q4_K_M and at Q8_0 are two files, two hashes,
and one base. llama.cpp writes that identity into every GGUF's metadata —
`general.architecture`, `general.basename`, `general.size_label` — and the
quantization into `general.file_type`. Reading those is a few hundred bytes
from the front of a multi-gigabyte file, so a validator can afford to do it
for every installed base on every import.

Standard library only. Values are read, never trusted beyond what they are:
a string in a header is a claim about the file, and the sha256 is what says
two files are the same file.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

_MAGIC = b"GGUF"
_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
           6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_STRING, _ARRAY = 8, 9

# llama.cpp's file_type enum, the ones the family actually ships. Anything
# else is reported by number rather than guessed.
_FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
               10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
               16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 30: "BF16"}


class NotGGUF(ValueError):
    pass


def read_header_bytes(data: bytes, *, max_array: int = 64) -> dict:
    """`read_header` for a GGUF already in memory — the adapter inside a
    `.pigeon` is read this way (`adapter.type`, spec §7.3 step 5). A truncated
    or foreign header raises NotGGUF rather than a struct error."""
    import io
    try:
        return _read(io.BytesIO(data), "adapter.lora.gguf", max_array)
    except (struct.error, KeyError, UnicodeDecodeError, MemoryError) as e:
        raise NotGGUF(f"adapter.lora.gguf has an unreadable GGUF header ({e})") from e


def read_header(path: Path | str, *, max_array: int = 64) -> dict:
    """Every key/value in the header. Long arrays (vocabularies) are cut to
    `max_array` entries — the identity fields are scalars and short lists."""
    with open(path, "rb") as f:
        return _read(f, path, max_array)


def _read(f, path, max_array: int) -> dict:
    """The header walk shared by the file and in-memory readers."""
    rd = f.read
    if rd(4) != _MAGIC:
        raise NotGGUF(f"{path} is not a GGUF file")
    version, = struct.unpack("<I", rd(4))
    if version < 2:
        raise NotGGUF(f"{path} is GGUF v{version}; v2+ is the readable layout")
    n_tensors, n_kv = struct.unpack("<QQ", rd(16))

    def string() -> str:
        n, = struct.unpack("<Q", rd(8))
        return rd(n).decode("utf-8", "replace")

    def value(t: int):
        if t == _STRING:
            return string()
        if t == _ARRAY:
            et, = struct.unpack("<I", rd(4))
            n, = struct.unpack("<Q", rd(8))
            out = []
            for i in range(n):
                v = value(et)
                if i < max_array:
                    out.append(v)
            return out
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, rd(size))[0]

    kv = {"_version": version, "_n_tensors": n_tensors}
    for _ in range(n_kv):
        key = string()
        t, = struct.unpack("<I", rd(4))
        kv[key] = value(t)
    return kv


def quant_name(header: dict) -> str:
    ft = header.get("general.file_type")
    if ft is None:
        return "unknown"
    return _FILE_TYPES.get(int(ft), f"file_type={ft}")


def base_identity(header: dict) -> dict:
    """The part of a base that stays the same across quantizations.

    `arch` is what the adapter's tensors are shaped for; `name` and `size` say
    which model of that architecture. Two files that agree on all three are
    the same base at possibly different quantizations — which is the
    documented, working case (FLIGHTSCHOOLSTATE.md §9: an adapter works "at any
    quantization of that same model").
    """
    name = (header.get("general.basename") or header.get("general.name") or "").strip()
    return {
        "arch": str(header.get("general.architecture") or "").strip(),
        "name": name,
        "size": str(header.get("general.size_label") or "").strip(),
        # Spec §7.2 (Draft 3, D12): Qwen3-VL-2B Instruct and Thinking share
        # arch, basename and size but not weights; only `general.finetune`
        # ("instruct" / "thinking") tells them apart. None when the header has none.
        "finetune": str(header.get("general.finetune") or "").strip() or None,
        "repo": str(header.get("general.base_model.0.repo_url") or header.get("general.repo_url") or "").strip() or None,
    }


def same_base(a: dict, b: dict) -> bool:
    """Identity match (spec §7.2), `a` the Pigeon's base, `b` an installed one.
    Case-insensitive, and tolerant of the spacing/hyphen noise different
    converters put in `general.name` ("Qwen2.5 1.5B Instruct" against
    "Qwen2.5-1.5B-Instruct"). An empty arch, name or size never matches."""
    norm = lambda s: "".join(ch for ch in str(s or "").lower() if ch.isalnum())
    if not a.get("arch") or a["arch"] != b.get("arch"):
        return False
    if norm(a.get("size")) != norm(b.get("size")):
        return False
    if not norm(a.get("size")):
        return False
    na, nb = norm(a.get("name")), norm(b.get("name"))
    # One converter writes the basename ("Qwen2.5"), another the full name
    # ("Qwen2.5 1.5B Instruct"); prefix agreement plus the size label is the
    # identity, not string equality.
    if not (na and nb and (na.startswith(nb) or nb.startswith(na))):
        return False
    # Spec §7.2 rule 4: when the Pigeon names a finetune, the base must carry an
    # equal one ("Instruct" and "instruct" agree). A base whose header has none
    # does not match. A Pigeon written before Draft 3 names none: rule skipped.
    if a.get("finetune"):
        return bool(norm(b.get("finetune"))) and norm(b.get("finetune")) == norm(a["finetune"])
    return True


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
