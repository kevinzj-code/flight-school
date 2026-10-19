"""The seal — who packed a `.pigeon`, checked before anything inside it is used.

docs/PIGEON-FORMAT.md, "Seal". A sealed file carries `seal.json` as its LAST
zip entry: an Ed25519 signature, by a key the app already trusts, over the
sha256 of every other entry. A `skillpack` (the one kind that ships `code/`)
must carry one; every other kind may, and if it does the seal must verify.

A seal proves who packed a thing, never that it is good.

The zip is parsed here byte by byte rather than through `zipfile`, because the
point of the strict rules is that every reader — this one, the Rust one and the
Node one — sees exactly the same entries or refuses the whole file. `zipfile`
is forgiving in ways (prepended bytes, trailing junk, a local header that
disagrees with the central directory) that would let two readers disagree
about what was signed. `zipfile` is used only to WRITE.

Standard library only; Ed25519 is RFC 8032 in pure Python. Slow, and fine: a
signature is checked once per import.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import stat
import struct
import zipfile
from pathlib import Path

SCHEME = "pigeon-seal-ed25519-v1"
SEAL_NAME = "seal.json"
MESSAGE_HEADER = b"PIGEON-SEAL-V1\n"
MAX_ENTRIES = 4096
MAX_NAME = 255

# The repositories a signing key must never be written into: the one this file
# lives in, plus any named in PIGEON_REPO_ROOTS (colon-separated). Never a
# hard-coded home folder — this file ships inside the app.
REPO_ROOTS = tuple(dict.fromkeys(
    [Path(__file__).resolve().parents[3]]
    + [Path(p).expanduser() for p in os.environ.get("PIGEON_REPO_ROOTS", "").split(":") if p.strip()]
))

_NAME = re.compile(rb"[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*")
_HEX16 = re.compile(r"[0-9a-f]{16}")
_HEX128 = re.compile(r"[0-9a-f]{128}")


class SealError(Exception):
    """Refused. `reason` is the one line an app shows."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Ed25519, RFC 8032 §5.1 (pure, no prehash, no context)

_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = t1 * 2 * _D * t2 % _P
    d = z1 * 2 * z2 % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(k: int, p):
    q = (0, 1, 1, 0)
    while k:
        if k & 1:
            q = _add(q, p)
        p = _add(p, p)
        k >>= 1
    return q


def _recover_x(y: int, sign: int):
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _encode(p) -> bytes:
    x, y, z, _ = p
    zi = _inv(z)
    x, y = x * zi % _P, y * zi % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decode(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)          # refuses y >= p: non-canonical encodings
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _h(*parts: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"".join(parts)).digest(), "little")


def _expand(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise ValueError("an Ed25519 secret seed is 32 bytes")
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(seed: bytes) -> bytes:
    a, _ = _expand(seed)
    return _encode(_mul(a, _G))


def sign(seed: bytes, msg: bytes) -> bytes:
    a, prefix = _expand(seed)
    pub = _encode(_mul(a, _G))
    r = _h(prefix, msg) % _L
    big_r = _encode(_mul(r, _G))
    k = _h(big_r, pub, msg) % _L
    s = (r + k * a) % _L
    return big_r + s.to_bytes(32, "little")


def verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    """Cofactorless check [S]B == R + [k]A, with S < L enforced and A and R
    required to decode canonically."""
    if len(pub) != 32 or len(sig) != 64:
        return False
    a = _decode(pub)
    r = _decode(sig[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _L:
        return False
    k = _h(sig[:32], pub, msg) % _L
    lhs = _mul(s, _G)
    rhs = _add(r, _mul(k, a))
    x1, y1, z1, _ = lhs
    x2, y2, z2, _ = rhs
    return (x1 * z2 - x2 * z1) % _P == 0 and (y1 * z2 - y2 * z1) % _P == 0


def key_id(pub: bytes) -> str:
    return hashlib.sha256(pub).hexdigest()[:16]


# ---------------------------------------------------------------------------
# the strict zip reader

_EOCD = struct.Struct("<IHHHHIIH")               # 22 bytes
_CEN = struct.Struct("<IHHHHHHIIIHHHHHII")       # 46 bytes
_LOC = struct.Struct("<IHHHHHIIIHH")             # 30 bytes


def _parse(data: bytes) -> list[dict]:
    """Every rule in the spec's STRICT ZIP RULES, or SealError."""
    n = len(data)
    if n < _EOCD.size:
        raise SealError("not a zip: too short for an end-of-central-directory record")
    (sig, disk, cd_disk, count_disk, count, cd_size, cd_off, comment_len) = _EOCD.unpack_from(data, n - 22)
    if sig != 0x06054B50:
        raise SealError("the end-of-central-directory record is not at the end of the file "
                        "(a comment or trailing bytes)")
    if comment_len != 0:
        raise SealError("the zip carries a comment")
    if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        raise SealError("zip64 is not allowed")
    if disk != 0 or cd_disk != 0 or count_disk != count:
        raise SealError("multi-disk zips are not allowed")
    if count > MAX_ENTRIES:
        raise SealError(f"more than {MAX_ENTRIES} entries")
    if cd_off + cd_size != n - 22:
        raise SealError("the central directory does not end where the end record starts")

    entries: list[dict] = []
    seen: set[bytes] = set()
    seen_lower: set[bytes] = set()
    pos, expect_local = cd_off, 0
    cd_end = cd_off + cd_size
    for _ in range(count):
        if pos + _CEN.size > cd_end:
            raise SealError("the central directory is shorter than its entry count")
        (csig, _made, _need, flags, method, mtime, mdate, _crc, csize, usize,
         nlen, xlen, clen, dstart, _iattr, _eattr, lho) = _CEN.unpack_from(data, pos)
        if csig != 0x02014B50:
            raise SealError("a central directory record has the wrong signature")
        name = data[pos + 46:pos + 46 + nlen]
        pos += 46 + nlen + xlen + clen
        if pos > cd_end:
            raise SealError("a central directory record runs past the directory")
        _check_name(name)
        if name in seen:
            raise SealError(f"duplicate entry {name.decode()!r}")
        if name.lower() in seen_lower:
            raise SealError(f"entry {name.decode()!r} collides with another when case is ignored")
        seen.add(name)
        seen_lower.add(name.lower())
        if flags not in (0, 0x800):
            raise SealError(f"entry {name.decode()!r} has general-purpose flags {flags:#x}")
        if method != 0:
            raise SealError(f"entry {name.decode()!r} is compressed; every entry must be stored")
        if 0xFFFFFFFF in (csize, usize, lho):
            raise SealError("zip64 is not allowed")
        if csize != usize:
            raise SealError(f"entry {name.decode()!r}: compressed size differs from size")
        if dstart != 0:
            raise SealError("multi-disk zips are not allowed")
        if lho != expect_local:
            raise SealError(f"entry {name.decode()!r} does not start where the previous one ended")
        if lho + _LOC.size > cd_off:
            raise SealError(f"entry {name.decode()!r}: local header overlaps the central directory")
        (lsig, _lver, lflags, lmethod, _lt, _ld, _lcrc, lcsize, lusize,
         lnlen, lxlen) = _LOC.unpack_from(data, lho)
        if lsig != 0x04034B50:
            raise SealError(f"entry {name.decode()!r}: no local header where the directory says")
        lname = data[lho + 30:lho + 30 + lnlen]
        if lname != name:
            raise SealError(f"entry {name.decode()!r}: local header name differs from the directory")
        if lflags not in (0, 0x800):
            raise SealError(f"entry {name.decode()!r} has general-purpose flags {lflags:#x}")
        if lmethod != method or lcsize != csize or lusize != usize:
            raise SealError(f"entry {name.decode()!r}: local header disagrees with the directory")
        start = lho + 30 + lnlen + lxlen
        end = start + csize
        if end > cd_off:
            raise SealError(f"entry {name.decode()!r} runs into the central directory")
        entries.append({"name": name.decode("ascii"), "data": data[start:end],
                        "date_time": _dos_time(mdate, mtime)})
        expect_local = end
    if pos != cd_end:
        raise SealError("the central directory size does not match its records")
    if expect_local != cd_off:
        raise SealError("bytes between the last entry and the central directory")
    return entries


def _check_name(name: bytes) -> None:
    shown = name.decode("latin-1")
    if not name or len(name) > MAX_NAME:
        raise SealError(f"entry name {shown!r} is empty or longer than {MAX_NAME} bytes")
    if not _NAME.fullmatch(name):
        raise SealError(f"entry name {shown!r} is not allowed")


def _dos_time(d: int, t: int) -> tuple:
    dt = ((d >> 9) + 1980, (d >> 5) & 0xF, d & 0x1F, t >> 11, (t >> 5) & 0x3F, (t & 0x1F) * 2)
    # zipfile refuses month/day 0; a hand-built zip may carry them.
    return (max(dt[0], 1980), max(dt[1], 1), max(dt[2], 1), dt[3], dt[4], dt[5])


#: The most a .pigeon or .nest may weigh. A picture Pigeon is under a
#: gigabyte (its projector is most of that); a nest of a few is a few. A file
#: past this is refused before it is read, so a stray 30 GB drop on the app
#: costs nothing.
MAX_FILE_BYTES = 8 * 1024 ** 3


def _bytes_of(path_or_bytes) -> bytes:
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        if len(path_or_bytes) > MAX_FILE_BYTES:
            raise SealError(f"file is {len(path_or_bytes) / 1024 ** 3:.1f} GB; the most a Pigeon may weigh is {MAX_FILE_BYTES // 1024 ** 3} GB")
        return bytes(path_or_bytes)
    p = Path(path_or_bytes)
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SealError(f"{p.name} is {size / 1024 ** 3:.1f} GB; the most a Pigeon may weigh is {MAX_FILE_BYTES // 1024 ** 3} GB")
    return p.read_bytes()


def strict_entries(path_or_bytes) -> list[tuple[str, bytes]]:
    return [(e["name"], e["data"]) for e in _parse(_bytes_of(path_or_bytes))]


# ---------------------------------------------------------------------------
# the seal

def seal_message(entries) -> bytes:
    """Every entry but seal.json, sorted by name bytes, as `sha256  name\\n`."""
    rows = sorted(((n.encode("ascii") if isinstance(n, str) else bytes(n), d)
                   for n, d in entries), key=lambda r: r[0])
    out = [MESSAGE_HEADER]
    for name, data in rows:
        if name == SEAL_NAME.encode():
            continue
        out.append(hashlib.sha256(data).hexdigest().encode() + b"  " + name + b"\n")
    return b"".join(out)


def _seal_json(entries_pairs, seed: bytes) -> bytes:
    pub = public_key(seed)
    sig = sign(seed, seal_message(entries_pairs))
    obj = {"scheme": SCHEME, "key": key_id(pub), "signature": sig.hex()}
    return json.dumps(obj, separators=(",", ":")).encode()


def build_sealed(entries, seed: bytes, date_times=None) -> bytes:
    """A sealed zip, in memory: `entries` in their order, seal.json last.
    Deterministic for fixed inputs (Ed25519 signatures are)."""
    pairs = [(n, bytes(d)) for n, d in entries]
    if any(n == SEAL_NAME for n, _ in pairs):
        raise SealError("already sealed — seal.json is present")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i, (name, data) in enumerate(pairs):
            dt = date_times[i] if date_times else (1980, 1, 1, 0, 0, 0)
            z.writestr(_info(name, dt), data)
        z.writestr(_info(SEAL_NAME, (1980, 1, 1, 0, 0, 0)), _seal_json(pairs, seed))
    return buf.getvalue()


def _info(name: str, dt) -> zipfile.ZipInfo:
    name.encode("ascii")                         # names must be ASCII
    zi = zipfile.ZipInfo(name, date_time=dt)
    zi.compress_type = zipfile.ZIP_STORED
    zi.flag_bits = 0
    zi.external_attr = 0o644 << 16
    return zi


def read_key(key_path) -> bytes:
    p = Path(key_path)
    st = p.stat()
    if st.st_mode & 0o077:
        raise SealError(f"{p} is readable by others (mode {stat.S_IMODE(st.st_mode):o}); "
                        "chmod 600 it before signing")
    text = p.read_text().strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise SealError(f"{p} is not a 64-hex-digit Ed25519 seed")
    return bytes.fromhex(text)


def seal_file(path, key_path) -> str:
    """Seal `path` in place. Returns the key id."""
    path = Path(path)
    seed = read_key(key_path)
    parsed = _parse(path.read_bytes())
    if any(e["name"] == SEAL_NAME for e in parsed):
        raise SealError("already sealed — seal.json is present")
    blob = build_sealed([(e["name"], e["data"]) for e in parsed], seed,
                        [e["date_time"] for e in parsed])
    part = path.with_name(path.name + ".part")
    part.write_bytes(blob)
    try:
        pub = public_key(seed)
        got = verify_file(part, {key_id(pub): pub.hex()})
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    os.replace(part, path)
    return got


def _read_seal(raw: bytes) -> dict:
    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise SealError("seal.json repeats a key")
        return dict(pairs)
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=no_dupes)
    except (ValueError, UnicodeDecodeError):
        raise SealError("seal.json is not valid JSON")
    if not isinstance(obj, dict) or set(obj) != {"scheme", "key", "signature"}:
        raise SealError("seal.json must have exactly scheme, key and signature")
    if obj["scheme"] != SCHEME:
        raise SealError(f"seal scheme {obj['scheme']!r} is not {SCHEME} — update the app")
    if not isinstance(obj["key"], str) or not _HEX16.fullmatch(obj["key"]):
        raise SealError("seal.json key must be 16 lowercase hex digits")
    if not isinstance(obj["signature"], str) or not _HEX128.fullmatch(obj["signature"]):
        raise SealError("seal.json signature must be 128 lowercase hex digits")
    return obj


def check_entries(entries, trusted: dict | None) -> str | None:
    """Key id for a valid seal, None for an unsealed file, SealError otherwise.
    `entries` must already have passed the strict rules."""
    names = [n for n, _ in entries]
    if SEAL_NAME not in names:
        return None
    if names.count(SEAL_NAME) != 1 or names[-1] != SEAL_NAME:
        raise SealError("seal.json must be the last entry, exactly once")
    seal = _read_seal(entries[-1][1])
    if not trusted:
        raise SealError("no trusted key — this app trusts no signer")
    pub_hex = trusted.get(seal["key"])
    if pub_hex is None:
        raise SealError(f"sealed by key {seal['key']}, which this app does not trust")
    pub = bytes.fromhex(pub_hex)
    if key_id(pub) != seal["key"]:
        raise SealError(f"trusted key list is wrong: {seal['key']} does not hash to that key")
    if not verify(pub, seal_message(entries[:-1]), bytes.fromhex(seal["signature"])):
        raise SealError("the seal does not verify — the file was changed after it was sealed")
    return seal["key"]


def check_file(path_or_bytes, trusted: dict | None) -> str | None:
    """Strict rules, then the seal. Key id, or None when unsealed."""
    return check_entries(strict_entries(path_or_bytes), trusted)


def verify_file(path_or_bytes, trusted: dict | None) -> str:
    """Like check_file, but an unsealed file is refused too."""
    got = check_file(path_or_bytes, trusted)
    if got is None:
        raise SealError("not sealed")
    return got


# ---------------------------------------------------------------------------
# keygen

def _inside_repo(p: Path) -> bool:
    s = str(p).lower()
    for root in REPO_ROOTS:
        r = str(root.resolve()).lower()
        if s == r or s.startswith(r.rstrip("/") + "/"):
            return True
    return False


def keygen(out_path) -> dict:
    out = Path(out_path).expanduser()
    resolved = out.resolve()
    if _inside_repo(resolved) or _inside_repo(out.absolute()):
        raise SealError(f"refusing to write a signing key inside a repository: {resolved}")
    if out.exists() or out.is_symlink():
        raise SealError(f"{out} already exists — refusing to overwrite a key")
    parent = out.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700)
        os.chmod(parent, 0o700)
    seed = secrets.token_bytes(32)
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (seed.hex() + "\n").encode())
    finally:
        os.close(fd)
    pub = public_key(seed)
    return {"path": str(out), "public_key": pub.hex(), "key": key_id(pub)}
