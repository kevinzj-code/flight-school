# Signing keys

Pigeon only runs code and adapters that are **sealed**: a `.pigeon` or `.nest`
carries `seal.json` as its last entry, an Ed25519 signature over the SHA-256 of
every other entry, made by a key the build trusts. Flight School is where seals
are made. Pigeon's own `docs/SIGNING.md` covers which keys a Pigeon build
trusts and how a fork trusts its own.

## What Flight School trusts

| File | Trusted to seal | Today |
|---|---|---|
| `src-tauri/keys/registry-keys.json` | the base-model registry (`src-tauri/registry/registry.zip`) | one key, id `25660ed562d33a0d` |

Only public keys live in this repository. The private keys were never
committed, and must never be.

Tests trust one published key, RFC 8032 §7.1 TEST 1, and nothing else.

## Making a key and sealing

The packer's CLI is `src-tauri/python/pigeon/cli.py`:

```bash
python3 src-tauri/python/pigeon/cli.py keygen --out ~/.airoplane-mode/seal.key
```

`keygen` writes the key with mode 0600 and refuses to write inside a
repository or over an existing file. Keep the key outside every repository and
back it up. If you lose it, you can't update anything it sealed. Anyone who
copies it can seal code that every build trusting it will run.

```bash
python3 src-tauri/python/pigeon/cli.py seal --file my.pigeon --key ~/.airoplane-mode/seal.key
python3 src-tauri/python/pigeon/cli.py verify --file my.pigeon --trusted-keys keys.json
```

`pack-skillpack` seals a skill pack the same way; Pigeon's `npm run pack:patch`
calls it. A key's id is the first 16 hex characters of the SHA-256 of its
public key, as `pigeon/seal.py` `key_id` computes it.

## Reporting a leaked key

If you believe a project key has been exposed, don't open a public issue. Use
the repository's private security advisory form. The fix is a release that
marks the key revoked and trusts a new one.
