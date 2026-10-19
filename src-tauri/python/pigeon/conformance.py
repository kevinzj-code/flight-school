#!/usr/bin/env python3
"""Flight School's adapter for the `.pigeon` conformance kit.

    python3 pigeon-format/conformance/run.py --impl "python3 <this file>"

The kit (`pigeon-format/conformance`, spec §16.2) calls this once per case with
the case folder and reads one JSON object back.

  check         Flight School's import check (`validate.py`) as a general Reader,
                with the host's trusted keys and, when the case carries one, its
                base registry (`registry.py`). When the file is `input.nest`:
                `nest.py`, then that same check on every member.
  registry      `registry.py`: the seal, then a base looked up by identity.
  prompt        `runner.messages` and `runner.read_answer` — the functions the
  answer        check ride builds its prompts and reads its answers with, because
                the ride is a Runner and its score must be one a host reproduces.
                A key press is `runner.action_for`. The host's lists for this
                ask (`host_values`, D86) must be for fields the file names.
  prompt_check  `promptproof.py`: the prompt through the base's chat template,
                compared with `prompt_check.canonical`. Needs jinja2.
  retrieve      `retrieve.py`, the Python port of the reference in `chunker.rs`.
  frames        `runner.frames_plan`: the recorder's capture size and crop, in
                Python (spec §13.2 steps 0 to 3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pigeon import embed, nest, promptproof, registry, retrieve, runner  # noqa: E402
from pigeon.manifest import SHOTS_FILE  # noqa: E402
from pigeon.seal import key_id, strict_entries  # noqa: E402
from pigeon.validate import validate_pigeon  # noqa: E402


def trusted(public_keys) -> dict:
    """The kit lists public keys; the app keeps {key id: public key}."""
    return {key_id(bytes.fromhex(p)): p for p in public_keys or []}


def open_registry(case_dir: Path, machine: dict | None) -> dict | None:
    """The registry beside the case, when its seal is one this host trusts for
    registries; one that is refused is simply not used (§7.4)."""
    path = case_dir / "registry.zip"
    if not path.exists():
        return None
    try:
        return registry.open_registry(path, trusted((machine or {}).get("registry_keys")))[0]
    except registry.RegistryRefused:
        return None


def check(pigeon: Path, machine: dict | None, reg: dict | None = None) -> dict:
    if machine is None:
        bases = projectors = None
        embedding_models: list | bool = False
        hashes: dict = {}
    else:
        bases, hashes = [], {}
        for i, b in enumerate(machine.get("bases", [])):
            path = f"<machine base {i}>"
            ident = b.get("identity") or {}
            bases.append({"path": path, "file": b.get("file"), "arch": ident.get("arch"),
                          "name": ident.get("name"), "size": ident.get("size"),
                          "finetune": ident.get("finetune"),
                          # a hash the host already knows (null: not hashed, D61)
                          **({"sha256": b["sha256"]} if b.get("sha256") else {})})
            hashes[path] = b.get("sha256")
        projectors = list(machine.get("projectors", []))
        embedding_models = list(machine.get("embedding_models", []))
    r = validate_pigeon(pigeon, bases, hash_check=lambda p: hashes.get(p),
                        trusted_keys=trusted((machine or {}).get("pigeon_keys")),
                        projectors=projectors, embedding_models=embedding_models,
                        registry=reg, general_reader=True)
    if not r.ok:
        out = {"result": "refused", "code": r.code, "detail": r.errors[0] if r.errors else ""}
        if r.needs is not None:
            out["needs"] = r.needs
        if r.pack_needs is not None:
            out["pack_needs"] = r.pack_needs
        return out
    return {"result": "accepted", "base_match": r.base_match, "warnings": sorted(r.warning_codes),
            "temperature": r.temperature, "self_test": r.self_test, "sealed_by": r.seal_key,
            "actions": r.actions}


def check_nest(case_dir: Path, machine: dict | None, reg: dict | None) -> dict:
    try:
        members = nest.unpack(case_dir / "input.nest")
    except nest.NotANest as e:
        return {"result": "refused", "code": e.code, "detail": str(e)}
    try:
        return {"result": "nest", "members": [{"file": m.name, **check(m, machine, reg)} for m in members]}
    finally:
        nest.cleanup(members)


def look_up(case_dir: Path, machine: dict | None, req: dict) -> dict:
    try:
        reg, key = registry.open_registry(case_dir / "registry.zip", trusted((machine or {}).get("registry_keys")))
    except registry.RegistryRefused as e:
        return {"result": "refused", "code": e.code, "detail": e.reason}
    out = {"result": "accepted", "sealed_by": key, "registry_version": reg.get("registry_version")}
    if "identity" in req:
        e = registry.entry_for(reg, req["identity"])
        offer = lambda row: row and {k: row.get(k) for k in ("file", "sha256", "bytes", "url")}  # noqa: E731
        out["base"] = e and {"id": e.get("id"), "license": e.get("license"), "license_url": e.get("license_url"),
                             "file": offer(registry.recommended(e.get("files"))),
                             "projector": offer(registry.recommended(e.get("projectors")))}
    return out


def ask(case_dir: Path, op: str, req: dict) -> dict:
    files = dict(strict_entries(case_dir / "input.pigeon"))
    m = json.loads(files["pigeon.json"])
    chunks = json.loads(files["material/chunks.json"])["chunks"] if "material/chunks.json" in files else []
    system = files["system_prompt.txt"].decode("utf-8") if "system_prompt.txt" in files else None
    shots = json.loads(files[SHOTS_FILE]) if SHOTS_FILE in files else None
    named = (m.get("output_contract") or {}).get("host_values") or []
    unnamed = [f for f in req.get("host_values") or {} if f not in named]
    if op in ("prompt", "answer") and unnamed:
        # D86: a host MUST NOT supply values for a field the file does not name.
        return {"result": "refused", "code": "host_values_unnamed", "detail": ", ".join(unnamed)}
    if op == "prompt":
        by_ordinal = {c["ordinal"]: c for c in chunks}
        sections = [by_ordinal[o] for o in req.get("sections", [])]
        out = {"messages": runner.messages(m, req["input"], system_prompt=system, sections=sections,
                                           history=req.get("history"), shots=shots)}
        if "chat_template" in req:
            out["rendered"] = promptproof.canonical(m, req["input"], req["chat_template"], system_prompt=system,
                                                    sections=sections, history=req.get("history"),
                                                    picture=req.get("picture"), shots=shots)
        return out
    if op == "prompt_check":
        got = promptproof.check(m, req["chat_template"], system_prompt=system, chunks=chunks,
                                picture=req.get("picture"), shots=shots)
        if not got["ok"]:
            return {"result": "refused", "code": "prompt_mismatch", "rendered": got["rendered"],
                    "position": got["position"]}
        return {"result": "accepted", "warnings": got["warnings"], "rendered": got["rendered"]}
    if op == "retrieve":
        method = m["material"]["retrieval"]
        units = dims = None
        if method == "hybrid-v1":
            emb = json.loads(files[embed.EMBEDDINGS_FILE])
            units, dims = retrieve.units_of(emb, chunks), emb["model"]["dims"]
        return {"picks": [retrieve.pick(method, chunks, q["question"], units=units,
                                        qvec=embed.unpack_f16(q["f16"], dims) if units is not None else None)
                          for q in req["questions"]]}
    labels = [l.strip() for l in files["labels.txt"].decode("utf-8").splitlines() if l.strip()] \
        if "labels.txt" in files else None
    payload = json.loads(files["skill/skill.json"]) if m.get("skill") and "skill/skill.json" in files else None
    out = runner.read_answer(m["output_contract"], req["raw"], labels=labels, kind=m.get("kind"),
                             skill_payload=payload, shown=req.get("shown"))
    if "actions.json" in files:
        out.update(runner.action_for(runner.usable_actions(files["actions.json"], m), out))
    if req.get("host_values"):
        out["host_values"] = req["host_values"]  # reported with the answer (D86)
    return out


def main(case_dir: str) -> dict:
    d = Path(case_dir)
    case = json.loads((d / "case.json").read_text())
    op, machine, req = case.get("op"), case.get("machine"), case.get("request") or {}
    if op == "check" and (d / "input.nest").exists():
        return check_nest(d, machine, open_registry(d, machine))
    if op == "check":
        return check(d / "input.pigeon", machine, open_registry(d, machine))
    if op == "registry":
        return look_up(d, machine, req)
    if op == "frames":
        return runner.frames_plan(req["window"], req["frames"])
    if op in ("prompt", "prompt_check", "retrieve", "answer"):
        return ask(d, op, req)
    return {"result": "unsupported", "detail": f"unknown op {op!r}"}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), ensure_ascii=False))
