"""The seal and the strict zip reader — docs/PIGEON-FORMAT.md §12.

The Rust and Node readers implement the same text; this file and the fixture
from make_seal_fixture.py are what keep the three in agreement.

    python3 -m pytest src-tauri/python/bench/test_seal.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from pigeon import seal, validate_pigeon  # noqa: E402
from pigeon.seal import SealError  # noqa: E402

from bench import make_seal_fixture as fixture  # noqa: E402
from bench.test_pigeon import base_q4, installed_bases, pack_run, run_folder  # noqa: E402

TEST_SEED = fixture.TEST_SEED
TEST_PUB = fixture.TEST_PUB
TEST_KEY = seal.key_id(TEST_PUB)
TRUSTED = {TEST_KEY: TEST_PUB.hex()}


# ---------------------------------------------------------------------------
# RFC 8032 §7.1

VECTORS = [
    # TEST 1
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
     "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    # TEST 2
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
     "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    # TEST 3
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
     "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    # TEST SHA(abc)
    ("833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42",
     "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
     hashlib.sha512(b"abc").hexdigest(),
     "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704"),
]


@pytest.mark.parametrize("seed,pub,msg,sig", VECTORS)
def test_rfc8032_vectors(seed, pub, msg, sig):
    seed, pub, msg, sig = (bytes.fromhex(x) for x in (seed, pub, msg, sig))
    assert seal.public_key(seed) == pub
    assert seal.sign(seed, msg) == sig
    assert seal.verify(pub, msg, sig)
    assert not seal.verify(pub, msg + b"x", sig)
    bad = bytearray(sig); bad[0] ^= 1
    assert not seal.verify(pub, msg, bytes(bad))


def test_non_canonical_s_is_refused():
    seed, pub, msg, sig = (bytes.fromhex(x) for x in VECTORS[0])
    s = int.from_bytes(sig[32:], "little") + seal._L
    assert s < 2 ** 256
    assert not seal.verify(pub, msg, sig[:32] + s.to_bytes(32, "little"))


def test_a_public_key_that_does_not_decode_is_refused():
    _, _, msg, sig = (bytes.fromhex(x) for x in VECTORS[0])
    not_canonical = (2 ** 255 - 19 + 1).to_bytes(32, "little")   # y >= p
    assert not seal.verify(not_canonical, msg, sig)


def test_the_test_key_id():
    assert TEST_KEY == hashlib.sha256(TEST_PUB).hexdigest()[:16]
    assert seal.public_key(TEST_SEED) == TEST_PUB


# ---------------------------------------------------------------------------
# the message

def test_seal_message_bytes_are_exact():
    entries = [("b.txt", b"B"), ("a/z.txt", b""), ("A.txt", b"a"), ("seal.json", b"{}")]
    h = lambda b: hashlib.sha256(b).hexdigest()
    want = (b"PIGEON-SEAL-V1\n"
            + f"{h(b'a')}  A.txt\n".encode()
            + f"{h(b'')}  a/z.txt\n".encode()
            + f"{h(b'B')}  b.txt\n".encode())
    assert seal.seal_message(entries) == want


def test_the_fixture_is_deterministic_and_verifies():
    blob1, msg1 = fixture.build()
    blob2, msg2 = fixture.build()
    assert blob1 == blob2 and msg1 == msg2
    h = lambda b: hashlib.sha256(b).hexdigest()
    assert msg1 == (b"PIGEON-SEAL-V1\n"
                    + f"{h(fixture.ENTRIES[1][1])}  code/skills/index.js\n".encode()
                    + f"{h(fixture.ENTRIES[0][1])}  pigeon.json\n".encode()
                    + f"{h(fixture.ENTRIES[2][1])}  prompts/planner.json\n".encode())
    names = [n for n, _ in seal.strict_entries(blob1)]
    assert names == ["pigeon.json", "code/skills/index.js", "prompts/planner.json", "seal.json"]
    assert seal.verify_file(blob1, TRUSTED) == TEST_KEY
    s = json.loads(dict(seal.strict_entries(blob1))["seal.json"])
    assert list(s) == ["scheme", "key", "signature"] and s["key"] == TEST_KEY


# ---------------------------------------------------------------------------
# a raw zip writer, so every rule can be broken on purpose

def raw_zip(entries, *, flags=0, method=0, csize_delta=0, local_names=None,
            gap=b"", prepend=b"", trailing=b"", comment=b""):
    body, central = bytearray(prepend), bytearray()
    for i, (name, data) in enumerate(entries):
        name = name if isinstance(name, bytes) else name.encode()
        lname = (local_names or {}).get(i, name)
        off = len(body)
        crc = zipfile.crc32(data) & 0xFFFFFFFF if hasattr(zipfile, "crc32") else __import__("zlib").crc32(data)
        size = len(data)
        body += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, flags, method, 0, 0x21,
                            crc, size + csize_delta, size, len(lname), 0) + lname + data
        central += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, flags, method, 0, 0x21,
                               crc, size + csize_delta, size, len(name), 0, 0, 0, 0, 0, off) + name
        if i == 0:
            body += gap
    cd_off = len(body)
    return bytes(body + central + struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries),
                                              len(central), cd_off, len(comment)) + comment + trailing)


GOOD = [("pigeon.json", b"{}"), ("a/b.txt", b"hello")]


def test_the_raw_writer_is_strictly_valid():
    assert seal.strict_entries(raw_zip(GOOD)) == GOOD


@pytest.mark.parametrize("name", [
    b"a\nb", b"..", b"a/../b", b"./x", b"a\\b", b"a\x00b", b"/abs", b"dir/", b"caf\xc3\xa9",
    b"x" * 256, b".hidden", b"", b"a//b",
])
def test_bad_names_are_refused(name):
    with pytest.raises(SealError):
        seal.strict_entries(raw_zip([("pigeon.json", b"{}"), (name, b"x")]))


def test_a_255_byte_name_is_allowed():
    seal.strict_entries(raw_zip([("x" * 255, b"x")]))


def test_duplicates_and_case_collisions_are_refused():
    with pytest.raises(SealError, match="duplicate"):
        seal.strict_entries(raw_zip([("a.txt", b"1"), ("a.txt", b"2")]))
    with pytest.raises(SealError, match="case"):
        seal.strict_entries(raw_zip([("README.md", b"1"), ("readme.md", b"2")]))


@pytest.mark.parametrize("kw", [
    {"gap": b"\x00"},
    {"prepend": b"PK"},
    {"trailing": b"\x00"},
    {"comment": b"hi"},
    {"flags": 0x8},
    {"method": 8},
    {"csize_delta": 1},
    {"local_names": {1: b"a/c.txt"}},
])
def test_layout_violations_are_refused(kw):
    with pytest.raises(SealError):
        seal.strict_entries(raw_zip(GOOD, **kw))


def test_flag_0x800_is_allowed():
    seal.strict_entries(raw_zip(GOOD, flags=0x800))


def test_zipfile_deflate_is_refused(tmp_path):
    p = tmp_path / "d.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pigeon.json", "{}" * 100)
    with pytest.raises(SealError):
        seal.strict_entries(p)


# ---------------------------------------------------------------------------
# sealing

def key_file(tmp_path, seed=TEST_SEED) -> Path:
    k = tmp_path / "keys" / "test.key"
    k.parent.mkdir(exist_ok=True)
    k.write_text(seed.hex() + "\n")
    os.chmod(k, 0o600)
    return k


def unsealed(tmp_path, entries=GOOD) -> Path:
    p = tmp_path / "u.pigeon"
    p.write_bytes(raw_zip(entries))
    return p


def test_seal_round_trip(tmp_path):
    p = unsealed(tmp_path)
    assert seal.seal_file(p, key_file(tmp_path)) == TEST_KEY
    got = seal.strict_entries(p)
    assert got[:-1] == GOOD and got[-1][0] == "seal.json"
    assert seal.verify_file(p, TRUSTED) == TEST_KEY
    assert not (tmp_path / "u.pigeon.part").exists()
    with pytest.raises(SealError, match="already sealed"):
        seal.seal_file(p, key_file(tmp_path))


def test_unsealed_is_reported(tmp_path):
    p = unsealed(tmp_path)
    assert seal.check_file(p, TRUSTED) is None
    with pytest.raises(SealError, match="not sealed"):
        seal.verify_file(p, TRUSTED)


def _sealed_entries(tmp_path):
    p = unsealed(tmp_path)
    seal.seal_file(p, key_file(tmp_path))
    return seal.strict_entries(p)


def test_tampering_breaks_the_seal(tmp_path):
    entries = _sealed_entries(tmp_path)
    body, s = entries[:-1], entries[-1]

    flipped = [(n, bytes([d[0] ^ 1]) + d[1:]) if n == "a/b.txt" else (n, d) for n, d in body]
    added = body + [("extra.txt", b"x")]
    renamed = [("a/c.txt", d) if n == "a/b.txt" else (n, d) for n, d in body]
    for bad in (flipped, added, renamed):
        with pytest.raises(SealError, match="does not verify"):
            seal.verify_file(raw_zip(bad + [s]), TRUSTED)

    obj = json.loads(s[1])
    obj["key"] = "0" * 16
    swapped = ("seal.json", json.dumps(obj, separators=(",", ":")).encode())
    with pytest.raises(SealError, match="does not trust"):
        seal.verify_file(raw_zip(body + [swapped]), TRUSTED)


def test_a_seal_by_another_key_fails(tmp_path):
    other = bytes(range(32))
    other_pub = seal.public_key(other)
    p = unsealed(tmp_path)
    seal.seal_file(p, key_file(tmp_path, other))
    with pytest.raises(SealError, match="does not trust"):
        seal.verify_file(p, TRUSTED)
    # a trusted list that maps the other key's id to the TEST key is caught
    with pytest.raises(SealError):
        seal.verify_file(p, {seal.key_id(other_pub): TEST_PUB.hex()})
    with pytest.raises(SealError, match="no trusted key"):
        seal.verify_file(p, {})


def test_seal_json_not_last_is_refused(tmp_path):
    entries = _sealed_entries(tmp_path)
    with pytest.raises(SealError, match="last"):
        seal.check_file(raw_zip([entries[-1]] + entries[:-1]), TRUSTED)


def test_a_loose_permission_key_is_refused(tmp_path):
    k = key_file(tmp_path)
    os.chmod(k, 0o644)
    with pytest.raises(SealError, match="readable by others"):
        seal.seal_file(unsealed(tmp_path), k)


# ---------------------------------------------------------------------------
# keygen

def test_keygen_writes_0600_and_never_overwrites(tmp_path):
    out = tmp_path / "newdir" / "signing.key"
    info = seal.keygen(out)
    assert (out.stat().st_mode & 0o777) == 0o600
    assert (out.parent.stat().st_mode & 0o777) == 0o700
    seed = bytes.fromhex(out.read_text().strip())
    assert out.read_text().endswith("\n") and len(out.read_text()) == 65
    assert info["public_key"] == seal.public_key(seed).hex()
    assert info["key"] == seal.key_id(seal.public_key(seed))
    with pytest.raises(SealError, match="already exists"):
        seal.keygen(out)


@pytest.mark.parametrize("root", seal.REPO_ROOTS)
def test_keygen_refuses_inside_either_repo(root):
    target = root / "never-written" / "signing.key"
    with pytest.raises(SealError, match="inside a repository"):
        seal.keygen(target)
    assert not target.exists() and not target.parent.exists()


# ---------------------------------------------------------------------------
# validate_pigeon: kinds and seals

def skillpack(tmp_path, *, sealed=True, manifest=None, extra=()) -> Path:
    m = manifest or json.loads(fixture.ENTRIES[0][1])
    entries = [("pigeon.json", json.dumps(m).encode())] + fixture.ENTRIES[1:] + list(extra)
    p = tmp_path / "pack.pigeon"
    p.write_bytes(seal.build_sealed(entries, TEST_SEED) if sealed else raw_zip(entries))
    return p


def test_a_sealed_skillpack_validates_with_the_trusted_key(tmp_path):
    r = validate_pigeon(skillpack(tmp_path), trusted_keys=TRUSTED)
    assert r.ok, r.errors
    assert r.seal_key == TEST_KEY


def test_the_fixture_validates(tmp_path):
    p = tmp_path / "f.pigeon"; p.write_bytes(fixture.build()[0])
    assert validate_pigeon(p, trusted_keys=TRUSTED).ok


def test_a_sealed_skillpack_with_no_trusted_keys_is_refused(tmp_path):
    r = validate_pigeon(skillpack(tmp_path))
    assert not r.ok and "no trusted key" in r.errors[0]


def test_an_unsealed_skillpack_is_refused(tmp_path):
    r = validate_pigeon(skillpack(tmp_path, sealed=False), trusted_keys=TRUSTED)
    assert not r.ok and "not sealed" in r.errors[0]


@pytest.mark.parametrize("edit,needle", [
    (lambda m: m.pop("models"), "models"),
    (lambda m: m.update(version=0), "version"),
    (lambda m: m.update(version=True), "version"),
    (lambda m: m.update(code={"entry": "code/missing.js"}), "not in the file"),
    (lambda m: m.update(code={"entry": "prompts/planner.json"}), "under code/"),
])
def test_skillpack_required_fields(tmp_path, edit, needle):
    m = json.loads(fixture.ENTRIES[0][1]); edit(m)
    r = validate_pigeon(skillpack(tmp_path, manifest=m), trusted_keys=TRUSTED)
    assert not r.ok and any(needle in e for e in r.errors), r.errors


def test_code_in_a_freeform_pigeon_is_refused_sealed_or_not(tmp_path):
    m = {"format_version": 1, "kind": "freeform", "name": "x", "id": "x"}
    entries = [("pigeon.json", json.dumps(m).encode()), ("code/x.js", b"1")]
    for blob in (raw_zip(entries), seal.build_sealed(entries, TEST_SEED)):
        p = tmp_path / "c.pigeon"; p.write_bytes(blob)
        r = validate_pigeon(p, trusted_keys=TRUSTED)
        assert not r.ok and "code/" in r.errors[0]


def _data_pigeon(tmp_path) -> Path:
    base = base_q4(tmp_path)
    out = tmp_path / "t.pigeon"
    pack_run(run_folder(tmp_path, base), out)
    return out


def test_a_data_only_pigeon_still_validates_unsealed(tmp_path):
    out = _data_pigeon(tmp_path)
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok, r.errors
    assert r.seal_key is None


def test_a_sealed_data_pigeon_must_verify(tmp_path):
    out = _data_pigeon(tmp_path)
    seal.seal_file(out, key_file(tmp_path))
    assert validate_pigeon(out, installed_bases([tmp_path]), trusted_keys=TRUSTED).seal_key == TEST_KEY
    # Spec §15.2: a data-only Pigeon sealed by a key this app does not trust is
    # reported, not refused — it is no less safe than the same file unsealed.
    r = validate_pigeon(out, installed_bases([tmp_path]))
    assert r.ok and "untrusted_seal" in r.warning_codes and "no trusted key" in r.warnings[0]
    entries = seal.strict_entries(out)
    tampered = [(n, d + b" ") if n == "README.md" else (n, d) for n, d in entries]
    out.write_bytes(raw_zip(tampered))
    r = validate_pigeon(out, installed_bases([tmp_path]), trusted_keys=TRUSTED)
    assert not r.ok and "does not verify" in r.errors[0]
