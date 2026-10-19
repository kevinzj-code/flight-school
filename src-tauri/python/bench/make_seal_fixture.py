"""The cross-app seal fixture: a tiny skillpack sealed with the RFC 8032 TEST 1
key, plus the exact message it signs. The Rust and Node readers check against
these bytes, so the build is deterministic (fixed dates, Ed25519 is
deterministic) and asserted so in bench/test_seal.py.

    python3 src-tauri/python/bench/make_seal_fixture.py --out-dir DIR
    -> DIR/sealed-test.pigeon, DIR/seal-message.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import seal  # noqa: E402

# RFC 8032 §7.1 TEST 1. Published; trusted only in tests.
TEST_SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
TEST_PUB = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")

ENTRIES = [
    ("pigeon.json", b'{"format_version":1,"name":"Test pack","id":"test-pack","kind":"skillpack",'
                    b'"version":1,"models":{},"code":{"entry":"code/skills/index.js"}}'),
    ("code/skills/index.js", b"export default [];\n"),
    ("prompts/planner.json", b'{"system":"hi"}'),
]


def build() -> tuple[bytes, bytes]:
    """(sealed zip bytes, signed message bytes)."""
    assert seal.public_key(TEST_SEED) == TEST_PUB
    return seal.build_sealed(ENTRIES, TEST_SEED), seal.seal_message(ENTRIES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    blob, msg = build()
    trusted = {seal.key_id(TEST_PUB): TEST_PUB.hex()}
    key = seal.verify_file(blob, trusted)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sealed-test.pigeon").write_bytes(blob)
    (out / "seal-message.txt").write_bytes(msg)
    print(json.dumps({"pigeon": str(out / "sealed-test.pigeon"), "bytes": len(blob),
                      "message": str(out / "seal-message.txt"), "key": key}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
