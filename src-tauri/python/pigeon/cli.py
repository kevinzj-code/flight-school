"""The packager, driven by Rust. Every reply is one JSON object on stdout.

    pack     --run-dir D --out F [--name N] [--readme-file R] [--icon I | --look JSON]
    look     --store DIR --name N [--line L] --llama-server B [--models DIR ...] [--timeout S]
             the Icon Maker's look for a Pigeon, or its random look and why
    look-kit --name N [--icon PNG]
             the name's random look, the parts to draw any look, and a chosen PNG
             [--example-seed S] [--example-count K]
    validate --file F [--models DIR ...]        the import check, as the apps run it
    add      --file F --store DIR [--models DIR ...] [--source S] [--run-id ID]
             a .pigeon or a .nest: validate each, copy the accepted into the lineup
    list     --store DIR                        the lineup, with what a card needs
    ask      --file F --text T --llama-server B [--models DIR ...] [--max-tokens N]
             USE one: import-check it, serve base+adapter, answer, and for a
             skill Pigeon render what the answer describes
    remove   --store DIR --id ID                to the Trash, never deleted outright
    base-pack --table D --out F [--store S] [--icon I]
             a base Pigeon (D83): a base with a system prompt, no training;
             rides the table's own rows (pigeon/basepack.py)
    nest pack --out F [--name N] PIGEON...      a flock in one file
    nest list --file F
    keygen   --out F                            a new Ed25519 signing key, mode 0600, never in a repo
    seal     --file F --key K                   append seal.json, signed by K
    verify   --file F [--trusted-keys J]        strict zip rules + seal; J is {key id: public hex}
    registry                                    open the app's registry: what it lists, or why not
    slim     --file F --out O                   rewrite F without its stock projector (§13.1)
    parts    --run-dir D [--models DIR ...]     what pack would put in the file, by the packer's own rule

Before the command, Rust names what the app trusts and has (pigeon/host.py):
    --registry Z --registry-keys K --hash-cache H
the sealed base registry it ships (D73), the key list for registries, and the
app's one cache of model hashes. A registry that does not open is logged on
stderr and not used: every projector is then carried.

`pack` validates the file it just wrote before reporting success: a run that
produces a .pigeon its own importer rejects is a failed run (spec §3.5).
"""

from __future__ import annotations

import argparse
import json
import zipfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import pack_run, validate_pigeon  # noqa: E402
from pigeon.validate import installed_bases  # noqa: E402
from pigeon import store as store_mod, nest as nest_mod  # noqa: E402
from pigeon import seal as seal_mod  # noqa: E402
from pigeon import skillpack as skillpack_mod  # noqa: E402
from pigeon import host as host_mod  # noqa: E402
from pigeon.pack import drop_stock_projector, file_parts  # noqa: E402
import pigeonlook  # noqa: E402


def _registry(a) -> dict | None:
    """The app's registry, opened once per command (6.5 KB), or None."""
    if not hasattr(a, "_registry"):
        a._registry = host_mod.open_bundled(a.registry, a.registry_keys)
    return a._registry


def _projectors(a) -> list[str] | None:
    """The projectors installed with the bases, by hash; None when the app
    has not said where its hash cache is (the check is then skipped)."""
    if not a.hash_cache:
        return None
    return host_mod.installed_projectors(a.models or [], a.hash_cache)


def _check(a, path):
    """The import check as this app runs it: with its registry and its
    installed projectors, as a fresh install would."""
    return validate_pigeon(path, installed_bases(a.models or []), trusted_keys=_trusted(a),
                           projectors=_projectors(a), registry=_registry(a))


def _card_args(a) -> dict:
    """The card's words, as the Package step or the command line gave them."""
    return {"does": getattr(a, "does", None), "input_hint": getattr(a, "input_hint", None),
            "output_noun": getattr(a, "output_noun", None), "talks_like": getattr(a, "talks_like", None)}


#: A table's own record of every Pigeon packed from it, by either button on
#: Send: {"packs": [{"revision", "file", "at"}]}, beside table.json.
PACKS_FILE = "packs.json"


def _packs(run_dir) -> list[dict]:
    try:
        rows = json.loads((Path(run_dir) / PACKS_FILE).read_text()).get("packs", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("revision"), int) and not isinstance(r["revision"], bool)]


def _produced_before(store, name: str | None, run_dir) -> int:
    """The highest revision a table produced before it kept its own record:
    the lineup's copy of its id (until 2026-10-06 both of Send's buttons added
    to the lineup, so that copy is its latest), or a Pigeon in its own export
    folder (where "Try it in Pigeon" writes). 0 when there is neither."""
    best = 0
    if store:
        from pigeon.manifest import build_manifest as _bm, slugify
        pid = slugify(name) if name else _bm(Path(run_dir), check_hash=False)[0]["id"]
        for row in store_mod.list_store(store):
            if row["id"] == pid:
                best = max(best, int(((row.get("card") or {}).get("revision")) or 1))
    for f in sorted((Path(run_dir) / "export").glob("*.pigeon")):
        try:
            with zipfile.ZipFile(f) as z:
                rev = (json.loads(z.read("pigeon.json")).get("card") or {}).get("revision")
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue
        if isinstance(rev, int) and not isinstance(rev, bool):
            best = max(best, rev)
    return best


def _next_revision(store, name: str | None, run_dir) -> int:
    """One above the table's own record of what it has packed, whichever
    button packed it, so no two files from one table share a number. A table
    with no record yet continues from what it produced before (_produced_before)."""
    packs = _packs(run_dir)
    if packs:
        return max(r["revision"] for r in packs) + 1
    return _produced_before(store, name, run_dir) + 1


def _record_pack(run_dir, revision: int, out) -> None:
    """Add a pack the import check accepted to the table's record."""
    from datetime import datetime, timezone
    path = Path(run_dir) / PACKS_FILE
    rows = _packs(run_dir) + [{"revision": revision, "file": str(out), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}]
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps({"packs": rows}, indent=2) + "\n")
    tmp.replace(path)


def cmd_look(a) -> dict:
    """A look for the Pigeon being packaged: the Icon Maker's, when the lineup
    has one and it answers with a valid look in time, else the name's random
    look — always with `source`, and a `reason` when it fell back. Never an
    error: the package step must open with an icon either way."""
    import iconrules
    fallback = {"recipe": pigeonlook.random_look(a.name), "source": "random"}
    found = store_mod.find_by_skill(a.store, pigeonlook.SKILL_NAME)
    if not found:
        return {**fallback, "reason": "no Icon Maker in the lineup"}
    from pigeon.ask import ask
    try:
        out = ask(found["path"], iconrules.icon_prompt(a.name, a.line or ""),
                  llama_server=a.llama_server, models=[Path(m) for m in (a.models or [])],
                  max_tokens=96, timeout_s=a.timeout)
    except Exception as e:  # noqa: BLE001 — any failure to answer means the random look, with why
        return {**fallback, "reason": f"the Icon Maker could not answer: {e}"}
    look, problems = pigeonlook.loads_look(out.get("answer") or "")
    if look is None:
        return {**fallback, "reason": "the Icon Maker's answer was not a look: " + "; ".join(problems)}
    return {"recipe": look, "source": "icon-maker", "pigeon": found["name"]}


def cmd_look_kit(a) -> dict:
    """What the Send screen draws an icon from: the name's random look, the
    parts table to draw any look with, and — when the person chose a PNG — that
    PNG as a data URL. No engine, no model: this is quick."""
    icon = None
    if a.icon and Path(a.icon).is_file():
        icon = store_mod.icon_data_url(Path(a.icon).read_bytes())
    return {"recipe": pigeonlook.random_look(a.name), "skill": pigeonlook.skill_payload(), "icon": icon}


def cmd_pack(a) -> dict:
    readme = Path(a.readme_file).read_text() if a.readme_file else None
    icon = Path(a.icon).read_bytes() if a.icon else None
    look = None
    if a.look and not icon:
        look, problems = pigeonlook.loads_look(a.look)
        if look is None:
            raise ValueError("the icon look is not one this app can draw: " + "; ".join(problems))
    warnings: list[str] = []
    revision = _next_revision(getattr(a, "store", None), a.name, a.run_dir)
    manifest = pack_run(a.run_dir, a.out, name=a.name, readme=readme, icon_png=icon, look=look,
                        example_seed=a.example_seed, example_count=a.example_count,
                        trained_with=a.trained_with, **_card_args(a),
                        revision=revision,
                        llama_server=getattr(a, "llama_server", None), model_dirs=a.models or [], warnings=warnings,
                        registry=_registry(a))
    # THE PACKAGER'S OWN GATE: re-open what was just written and pass it
    # through the same check the apps run — with the registry, so a file
    # without its stock projector must find it installed with the base.
    report = _check(a, a.out)
    if report.ok:
        # Only a file that is kept takes its number: one that fails the check
        # is deleted by the app, and its number is used again.
        _record_pack(a.run_dir, revision, a.out)
    out = Path(a.out)
    return {"path": str(out), "bytes": out.stat().st_size if out.exists() else None,
            "manifest": manifest, "validation": report.to_json(), "warnings": warnings}


def cmd_base_pack(a) -> dict:
    from instructor.config import load as load_config
    from pigeon import basepack
    from pigeon.hashes import HashCache
    table = json.loads((Path(a.table) / basepack.TABLE_FILE).read_text())
    revision = _next_revision(a.store, table["name"], a.table)
    hashes = HashCache(a.hash_cache)
    got = basepack.pack_base(a.table, a.out, cfg=load_config(Path(a.table) / "config.toml"), registry=_registry(a),
                             hashes=hashes, revision=revision,
                             icon_png=Path(a.icon).read_bytes() if a.icon else None)
    hashes.save()
    report = _check(a, a.out)                    # the packager's own gate, as for pack
    if report.ok:
        _record_pack(a.table, revision, a.out)
    out = Path(a.out)
    return {"path": str(out), "bytes": out.stat().st_size if out.exists() else None, "manifest": got["manifest"],
            "check_ride": {k: v for k, v in got["flight"].items() if k != "rows"}, "validation": report.to_json()}


def cmd_validate(a) -> dict:
    return _check(a, a.file).to_json()


def cmd_add(a) -> dict:
    results = store_mod.add(a.file, a.store, a.models or [], source=a.source, run_id=a.run_id,
                            registry=_registry(a), projectors=_projectors(a))
    return {"results": results,
            "added": sum(r["status"] == "added" for r in results),
            "already": sum(r["status"] == "already" for r in results),
            "refused": sum(r["status"] == "refused" for r in results)}


def cmd_list(a) -> dict:
    return {"pigeons": store_mod.list_store(a.store)}


def cmd_remove(a) -> dict:
    return store_mod.remove(a.store, a.id)


def cmd_nest(a) -> dict:
    if a.nest_cmd == "pack":
        return {"path": a.out, "nest": nest_mod.pack(a.out, a.pigeons, a.name),
                "bytes": Path(a.out).stat().st_size}
    return nest_mod.list_nest(a.file)


def _trusted(a) -> dict:
    """{key id: public key hex} from a JSON file, or nothing trusted."""
    if not getattr(a, "trusted_keys", None):
        return {}
    keys = json.loads(Path(a.trusted_keys).read_text())
    if not isinstance(keys, dict):
        raise ValueError("--trusted-keys must be a JSON object of key id to public key hex")
    return keys


def cmd_keygen(a) -> dict:
    return seal_mod.keygen(a.out)


def cmd_seal(a) -> dict:
    return {"path": a.file, "key": seal_mod.seal_file(a.file, a.key)}


def cmd_pack_skillpack(a) -> dict:
    return skillpack_mod.pack_skillpack(a.stage, a.key, a.out, allow_dirty=a.allow_dirty)


def cmd_verify(a) -> dict:
    """Strict rules and the seal. A bad seal is an error line and exit 2; an
    unsealed but well-formed file says so."""
    key = seal_mod.check_file(a.file, _trusted(a))
    return {"path": a.file, "sealed": key is not None, "key": key}


def cmd_registry(a) -> dict:
    """What the app's registry lists, or that it is not used — the launch log."""
    reg = _registry(a)
    if reg is None:
        return {"opened": False}
    return {"opened": True, "registry_version": reg.get("registry_version"),
            "bases": [b.get("id") for b in reg.get("bases", []) if isinstance(b, dict)]}


def cmd_slim(a) -> dict:
    """Rewrite a packed Pigeon without its stock projector, then run the
    import check on the result. A file that would fail it is not kept."""
    reg = _registry(a)
    out = Path(a.out)
    before = Path(a.file).stat().st_size
    done = drop_stock_projector(a.file, out, reg)
    if not done["dropped"]:
        return {**done, "path": a.file, "bytes": before}
    report = _check(a, out)
    if not report.ok:
        out.unlink(missing_ok=True)
        raise ValueError("the rewritten file failed its own import check, so it was not kept: "
                         + " · ".join(report.errors))
    return {**done, "path": str(out), "bytes_before": before, "bytes": out.stat().st_size,
            "validation": report.to_json()}


def cmd_parts(a) -> dict:
    return file_parts(a.run_dir, registry=_registry(a), hash_cache=a.hash_cache)


def cmd_ask(a) -> dict:
    """USE a Pigeon. The import check runs first, so a file this machine cannot
    load is refused before an engine is started."""
    from pigeon.ask import ask
    return ask(a.file, a.text, llama_server=a.llama_server,
               models=[Path(m) for m in (a.models or [])],
               ctx=a.ctx, max_tokens=a.max_tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry"); ap.add_argument("--registry-keys"); ap.add_argument("--hash-cache")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("--run-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--name"); p.add_argument("--readme-file"); p.add_argument("--icon"); p.add_argument("--look")
    p.add_argument("--example-seed", type=int, default=11); p.add_argument("--example-count", type=int, default=4)
    p.add_argument("--trained-with", default="Flight School"); p.add_argument("--models", nargs="*")
    p.add_argument("--store"); p.add_argument("--llama-server", help="to embed a Reader's material (hybrid-v1); without it the lexical search is packed")
    p.add_argument("--does"); p.add_argument("--input-hint"); p.add_argument("--output-noun")
    p.add_argument("--talks-like")
    lk = sub.add_parser("look"); lk.add_argument("--store", required=True); lk.add_argument("--name", required=True)
    lk.add_argument("--line", default=""); lk.add_argument("--llama-server", required=True)
    lk.add_argument("--models", nargs="*"); lk.add_argument("--timeout", type=float, default=60)
    lkk = sub.add_parser("look-kit"); lkk.add_argument("--name", required=True); lkk.add_argument("--icon")
    v = sub.add_parser("validate"); v.add_argument("--file", required=True); v.add_argument("--models", nargs="*")
    v.add_argument("--trusted-keys")
    kg = sub.add_parser("keygen"); kg.add_argument("--out", required=True)
    se = sub.add_parser("seal"); se.add_argument("--file", required=True); se.add_argument("--key", required=True)
    ps = sub.add_parser("pack-skillpack"); ps.add_argument("--stage", required=True); ps.add_argument("--key", required=True)
    ps.add_argument("--out", required=True); ps.add_argument("--allow-dirty", action="store_true")
    ve = sub.add_parser("verify"); ve.add_argument("--file", required=True); ve.add_argument("--trusted-keys")
    sub.add_parser("registry")
    sl = sub.add_parser("slim"); sl.add_argument("--file", required=True); sl.add_argument("--out", required=True)
    sl.add_argument("--models", nargs="*")
    pa = sub.add_parser("parts"); pa.add_argument("--run-dir", required=True); pa.add_argument("--models", nargs="*")
    ad = sub.add_parser("add"); ad.add_argument("--file", required=True); ad.add_argument("--store", required=True)
    ad.add_argument("--models", nargs="*"); ad.add_argument("--source", default="added"); ad.add_argument("--run-id")
    ls = sub.add_parser("list"); ls.add_argument("--store", required=True)
    ak = sub.add_parser("ask"); ak.add_argument("--file", required=True)
    ak.add_argument("--text", required=True); ak.add_argument("--llama-server", required=True)
    ak.add_argument("--models", nargs="*"); ak.add_argument("--max-tokens", type=int)
    ak.add_argument("--ctx", type=int, default=2048)
    rm = sub.add_parser("remove"); rm.add_argument("--store", required=True); rm.add_argument("--id", required=True)
    bp = sub.add_parser("base-pack"); bp.add_argument("--table", required=True); bp.add_argument("--out", required=True)
    bp.add_argument("--store"); bp.add_argument("--icon"); bp.add_argument("--models", nargs="*")
    ne = sub.add_parser("nest"); nsub = ne.add_subparsers(dest="nest_cmd", required=True)
    npk = nsub.add_parser("pack"); npk.add_argument("--out", required=True); npk.add_argument("--name"); npk.add_argument("pigeons", nargs="+")
    nls = nsub.add_parser("list"); nls.add_argument("--file", required=True)
    a = ap.parse_args()
    try:
        out = {"pack": cmd_pack, "look": cmd_look, "look-kit": cmd_look_kit, "validate": cmd_validate, "add": cmd_add,
               "list": cmd_list, "remove": cmd_remove, "nest": cmd_nest,
               "ask": cmd_ask, "keygen": cmd_keygen, "seal": cmd_seal,
               "verify": cmd_verify, "pack-skillpack": cmd_pack_skillpack,
               "registry": cmd_registry, "slim": cmd_slim, "parts": cmd_parts, "base-pack": cmd_base_pack}[a.cmd](a)
    except Exception as e:  # one honest line, never a traceback the UI has to parse
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 2
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
