"""The import check — what any app must do on opening a `.pigeon`.

Spec: `pigeon-format/SPEC.md` §7.3 (Version 1, Draft 4). The steps run in the
spec's order and stop at the first step that refuses:

  1. container   2. version   3. code and seal   4. fields   5. files by kind
  6. the base (and, for a picture Pigeon, its projector)

Every refusal and warning carries the stable code from the spec's Appendix A
(`ValidationReport.code`, `warning_codes`); the sentence beside it is ours and
names exactly what failed. Two rules from the older internal doc still hold and
are worth restating:

  * AN EXACT HASH IS A FAST PATH, NOT THE REFUSAL RULE. Identity refuses; a
    file of the same base at another quantization loads with a warning.
  * A SELF-TEST NEVER CLAIMS A VERDICT IT CANNOT GRADE (§12.2).

This module is Flight School's Reader. It is held to the spec by the
conformance kit (`pigeon-format/conformance`, via `pigeon/conformance.py`) and
to the apps by bench/test_pigeon.py. Standard library only, on purpose.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import seal as seal_mod
from .gguf import NotGGUF, base_identity, read_header, read_header_bytes, same_base
from .manifest import (CODE_KINDS, FORMAT_VERSION, KINDS, MATERIAL_SLOT, RETRIEVALS, SHOTS_FILE,
                       schema_field, shots_problem, skill_payload_problems)

BASE_EXACT = "exact"                    # the very file it was tested on
BASE_SAME_OTHER_QUANT = "same_base_other_quant"   # warn and load
BASE_MISSING = "missing"                # offer to download; never substitute
BASE_WRONG = "wrong"                    # refuse, name the model needed

LABEL_KINDS = ("classifier", "router")  # kinds whose self-test is always a verdict
OUTPUT_TYPES = ("label", "json", "text", "reasoned")
HISTORY_SHAPES = ("none", "as_sent")

# §5.2: what every non-skillpack manifest must have. `trained_on`, `trained_with`
# and `example_count` are optional in the spec; Flight School still writes them.
REQUIRED_BASE = ("file", "arch", "name", "size")
REQUIRED_FILES = {"classifier": ("labels.txt",), "router": ("labels.txt",),
                  "persona": ("system_prompt.txt",),
                  # A grounded Pigeon that does not carry its material cannot
                  # answer anything: the material is where its answers come from.
                  "grounded": ("material/chunks.json",)}
# A skillpack has no adapter, base or template; what it has is code and a seal.
REQUIRED_SKILLPACK = ("format_version", "name", "id", "kind", "version", "models", "code")


class _Stop(Exception):
    """A step refused; the report already holds the code and the sentence."""


def _version_of(v):
    """A skill version as Pigeon reads it: a whole number, or a string holding
    one (Flight School's manifest writes the number, its payload the string).
    Anything else — a bool, a float, a word — is no version at all."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _is_u32(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 0xFFFFFFFF


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def plain_id(i) -> bool:
    """§5.3: a plain file-name component, because hosts name files after it."""
    return (isinstance(i, str) and 1 <= len(i) <= 128 and not i.startswith(".")
            and all((c.isascii() and c.isalnum()) or c in "-_. " for c in i))


def _chunks_problems(raw: bytes) -> list[str]:
    """`material/chunks.json` checked the way Pigeon PARSES it
    (`material.rs::ChunkIndex`): an object with a non-empty `chunks` list, each
    chunk an integer `ordinal` and string `text`. The optional fields are typed
    too — serde refuses a `title` of null as surely as a missing `text`, so a file
    that only a lenient reader accepts would install here and fail in Pigeon."""
    try:
        idx = json.loads(raw)
    except ValueError as e:
        return [f"material/chunks.json could not be read — {e}"]
    if not isinstance(idx, dict) or not isinstance(idx.get("chunks"), list):
        return ["material/chunks.json must be an object with a 'chunks' list"]
    if not idx["chunks"]:
        return ["material/chunks.json has no sections in it"]
    if "total_est_tokens" in idx and not _is_u32(idx["total_est_tokens"]):
        return ["material/chunks.json total_est_tokens must be a whole number"]
    for i, c in enumerate(idx["chunks"]):
        if not isinstance(c, dict):
            return [f"material/chunks.json chunk {i} is not an object"]
        if not _is_u32(c.get("ordinal")):
            return [f"material/chunks.json chunk {i} needs a whole-number 'ordinal'"]
        if not isinstance(c.get("text"), str):
            return [f"material/chunks.json chunk {i} needs a string 'text'"]
        for key in ("title", "file"):
            if key in c and not isinstance(c[key], str):
                return [f"material/chunks.json chunk {i} '{key}' must be a string"]
        if "est_tokens" in c and not _is_u32(c["est_tokens"]):
            return [f"material/chunks.json chunk {i} 'est_tokens' must be a whole number"]
    return []


def _skill_payload_problems(raw: bytes, name, version) -> list[str]:
    """`skill/skill.json` parsed, then checked by `manifest.skill_payload_problems`
    — the one definition the packager checks a run's skill against, too."""
    try:
        p = json.loads(raw)
    except ValueError as e:
        return [f"skill/skill.json could not be read — {e}"]
    return skill_payload_problems(p, name, version)


def self_test_for(m: dict) -> str:
    """§12.2 (D31): pass/fail only where the answer is checked exactly — label
    kinds, label answers (inside `reasoned` too), and skill recipes."""
    oc = m.get("output_contract") or {}
    exact = oc.get("answer") or oc.get("type")
    if m.get("kind") in LABEL_KINDS or exact == "label" or (exact == "json" and m.get("skill")):
        return "verdict"
    return "samples"


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: dict | None = None
    files: list[str] = field(default_factory=list)
    base_match: str | None = None
    base_needed: str | None = None
    #: "verdict" or "samples" — what the importer may show for examples/ (§12.2)
    self_test: str | None = None
    examples: int = 0
    #: the key id of a seal that verified, or None when unsealed
    seal_key: str | None = None
    #: Appendix A.1 code of the refusal (the first error recorded), or None
    code: str | None = None
    #: Appendix A.2 codes, one per warning
    warning_codes: list[str] = field(default_factory=list)
    #: the temperature to run at after §5.5's rules
    temperature: float | None = None
    #: for wrong_base, no_base and no_projector: what a host needs to offer the
    #: download and show the licence first (§16.4) — `registry.needs`
    needs: dict | None = None
    #: the file's actions.json when it has one that was not skipped (§7.3 step 7)
    actions: dict | None = None
    #: a skillpack's `needs`, the Pigeons its code asks (D84), as listed
    pack_needs: list | None = None

    def error(self, msg: str, code: str = "bad_manifest") -> None:
        self.ok = False
        self.errors.append(msg)
        if self.code is None:
            self.code = code

    def warn(self, msg: str, code: str) -> None:
        self.warnings.append(msg)
        self.warning_codes.append(code)

    def to_json(self) -> dict:
        return {"ok": self.ok, "code": self.code, "errors": self.errors, "warnings": self.warnings,
                "warning_codes": self.warning_codes, "manifest": self.manifest, "files": self.files,
                "base_match": self.base_match, "base_needed": self.base_needed,
                "self_test": self.self_test, "temperature": self.temperature,
                "examples": self.examples, "seal_key": self.seal_key, "needs": self.needs,
                "actions": self.actions}


def installed_bases(model_dirs) -> list[dict]:
    """Every GGUF the importing app has, with its identity read from its
    header. `sha256` is left out: hashing every multi-gigabyte model on every
    import is not the fast path. The validator hashes one candidate only when
    the identity already matches."""
    out = []
    for d in model_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.gguf")):
            try:
                h = read_header(p)
            except Exception:
                continue
            if h.get("general.type", "model") != "model":
                continue
            out.append({"path": str(p), "file": p.name, **base_identity(h)})
    return out


def _validate_skillpack(r: ValidationReport, m: dict) -> ValidationReport:
    """A skillpack (§15.3, reserved) — Flight School seals Pigeon's own, so it
    reads them. No base, so there is no step 6."""
    for k in REQUIRED_SKILLPACK:
        if k not in m or m[k] in (None, ""):
            r.error(f"pigeon.json is missing '{k}'", "missing_field")
    for k in ("name", "id"):
        if k in m and not isinstance(m[k], str):
            r.error(f"'{k}' must be text")
    ver = m.get("version")
    if "version" in m and (isinstance(ver, bool) or not isinstance(ver, int) or ver < 1):
        r.error("'version' must be a whole number, at least 1")
    if "models" in m and not isinstance(m["models"], dict):
        r.error("'models' must be an object")
    code = m.get("code")
    if "code" in m and not isinstance(code, dict):
        r.error("'code' must be an object")
    elif isinstance(code, dict):
        entry = code.get("entry")
        if not isinstance(entry, str) or not entry.startswith("code/"):
            r.error("code.entry must name a file under code/")
        elif entry not in r.files:
            r.error(f"code.entry {entry!r} is not in the file", "missing_required_file")
    r.self_test = "samples"
    return r


def validate_pigeon(path: Path | str, bases: list[dict] | None = None, *,
                    hash_check=None, trusted_keys: dict | None = None,
                    projectors: list[str] | None = None,
                    embedding_models: list[str] | None | bool = None,
                    registry: dict | None = None, general_reader: bool = False) -> ValidationReport:
    """Spec §7.3 steps 1–6; copying, registering and running belong to the app.

    `bases` is the machine's installed models as `installed_bases` returns them,
    or None to check the file only. `hash_check(path) -> sha256` is injectable so
    tests do not hash real gigabytes. `trusted_keys` is {key id: public key hex}.
    `projectors` lists the projector hashes installed with the bases; None skips
    the projector-availability check (Flight School still carries projectors).
    `embedding_models` lists the hybrid-v1 embedding models this app holds;
    None means Flight School's own one, False skips the check.
    `registry` is a base registry already opened under a trusted registry key
    (`registry.open_registry`). A projector it lists for the Pigeon's base counts
    as available (§7.3 step 6), a spelling it accepts counts as the same base
    (D45), and an entry one of whose files disagrees with it is ignored, with a
    warning (D46). `general_reader` refuses a skillpack whose seal
    is good with `unsupported_kind` (§15.3); without it Flight School reads
    them, because it is the app that seals Pigeon's own.
    """
    r = ValidationReport()
    try:
        _check(r, Path(path), bases, hash_check, trusted_keys, projectors, embedding_models,
               registry, general_reader)
    except _Stop:
        pass
    return r


def _check(r, path, bases, hash_check, trusted_keys, projectors, embedding_models,
           registry=None, general_reader=False) -> None:
    from .gguf import sha256_file
    hash_check = hash_check or sha256_file

    # 1. Container: a strict ZIP (§3.2) with pigeon.json at the root.
    if not zipfile.is_zipfile(path):
        r.error(f"{path.name} is not a .pigeon — it is not a ZIP archive", "not_a_zip")
        raise _Stop
    try:
        entries = seal_mod.strict_entries(path)
    except seal_mod.SealError as e:
        r.error(f"refused: {e.reason}", "bad_zip")
        raise _Stop
    r.files = [n for n, _ in entries]
    by_name = dict(entries)
    if "pigeon.json" not in by_name:
        r.error("pigeon.json is missing — every .pigeon carries one at its root", "no_manifest")
        raise _Stop
    try:
        m = json.loads(by_name["pigeon.json"])
    except ValueError as e:
        r.error(f"pigeon.json is not valid JSON: {e}")
        raise _Stop
    if not isinstance(m, dict):
        r.error("pigeon.json must be a JSON object")
        raise _Stop
    r.manifest = m

    # 2. Version.
    v = m.get("format_version")
    if v is None:
        r.error("pigeon.json is missing 'format_version'", "missing_field")
        raise _Stop
    if not _whole(v):
        r.error("format_version is not a whole number")
        raise _Stop
    if v > FORMAT_VERSION:
        r.error(f"this file is .pigeon format {v}; this app reads up to {FORMAT_VERSION}. "
                f"Update the app to open it", "newer_format")
        raise _Stop

    # 3. Code and seal. Code travels only in a skillpack; a skillpack is only
    # ever opened under a trusted seal; any other seal must verify, and one by a
    # key this app does not trust is reported, not refused (§15.2).
    kind = m.get("kind")
    if kind not in CODE_KINDS and any(f.startswith("code/") for f in r.files):
        r.error(f"kind {kind!r} may not carry code/ — only a sealed skillpack ships code",
                "code_in_wrong_kind")
        raise _Stop
    if kind in CODE_KINDS or seal_mod.SEAL_NAME in r.files:
        try:
            r.seal_key = seal_mod.check_entries(entries, trusted_keys or {})
        except seal_mod.SealError as e:
            untrusted = "trust" in e.reason
            if kind in CODE_KINDS or not untrusted:
                r.error(f"refused: {e.reason}", "untrusted_seal" if untrusted else "broken_seal")
                raise _Stop
            r.warn(f"sealed, but {e.reason}", "untrusted_seal")
        if kind in CODE_KINDS and r.seal_key is None:
            r.error("a skillpack must be sealed by a trusted key, and this one is not sealed",
                    "code_without_seal")
            raise _Stop
    if kind in CODE_KINDS:
        needs = m.get("needs")
        if isinstance(needs, list) and all(isinstance(n, str) and plain_id(n) for n in needs):
            r.pack_needs = needs
        if general_reader:
            r.error(f"kind {kind!r} is sealed by a trusted key, but a general Reader does not run it",
                    "unsupported_kind")
            raise _Stop
        _validate_skillpack(r, m)
        return

    # 4. Fields.
    def need(obj, key, typ, where=""):
        if key not in obj or obj[key] is None:
            r.error(f"pigeon.json is missing '{where}{key}'", "missing_field")
            raise _Stop
        if not isinstance(obj[key], typ) or (typ is int and isinstance(obj[key], bool)):
            r.error(f"pigeon.json '{where}{key}' has the wrong type")
            raise _Stop
        return obj[key]

    need(m, "name", str)
    if not plain_id(need(m, "id", str)):
        r.error(f"id {m['id']!r} is not a plain file-name component (§5.3)", "bad_id")
        raise _Stop
    need(m, "kind", str)
    if kind not in KINDS:
        r.error(f"kind {kind!r} is not one of {', '.join(KINDS)}", "unknown_kind")
        raise _Stop
    bm = need(m, "base_model", dict)
    for k in REQUIRED_BASE:
        need(bm, k, str, "base_model.")
    # D83: a base Pigeon carries no trained weights.
    base_only = m.get("base_only", False)
    if not isinstance(base_only, bool):
        r.error("base_only must be true or false")
        raise _Stop
    # An empty name or size is allowed here and matches nothing (§7.2): the
    # Pigeon is refused at step 6 as wrong_base or no_base, and `needs` leaves
    # the empty field out of the identity (D63).
    template = need(m, "prompt_template", str)
    oc = need(m, "output_contract", dict)
    otype = need(oc, "type", str, "output_contract.")
    if not _whole(oc.get("max_tokens")):
        if "max_tokens" not in oc:
            r.error("pigeon.json is missing 'output_contract.max_tokens'", "missing_field")
        else:
            r.error("output_contract.max_tokens must be a whole number")
        raise _Stop
    inputs = need(m, "inputs", list)
    if "{input}" not in template:
        r.error("prompt_template must contain {input} — the runtime reproduces it byte for byte",
                "template_missing_input")
        raise _Stop
    if otype not in OUTPUT_TYPES:
        r.error(f"output_contract.type {otype!r} is not one this version reads "
                f"({', '.join(OUTPUT_TYPES)}) — update the app", "unknown_output_type")
        raise _Stop
    inner = oc.get("answer")
    if (otype == "reasoned") != (inner is not None) or (inner is not None and inner not in ("label", "json")):
        r.error("output_contract.answer goes with type 'reasoned' only, and is 'label' or 'json'")
    answer_type = inner or otype
    if kind in LABEL_KINDS and answer_type != "label":
        # D56: a router names where to send the input, so it answers with a label too.
        r.error(f"a {kind}'s output_contract.type must be 'label' (or 'reasoned' with a label answer)")
    if answer_type == "json" and m.get("skill") and not oc.get("fields"):
        r.error("output_contract.type is json but 'fields' is empty")
    constrained = oc.get("constrained", False)
    if not isinstance(constrained, bool) or (constrained and otype == "text"):
        r.error("output_contract.constrained must be true or false, and is not allowed with a text answer")
    # D86, D88, D89: fields whose allowed values the host supplies per ask, and
    # the JSON Schema they are fields of.
    schema = oc.get("output_schema")
    if schema is not None and (not isinstance(schema, dict) or answer_type != "json"):
        r.error("output_contract.output_schema is a JSON Schema object, and goes with a json answer")
    named = oc.get("host_values", [])
    if not (isinstance(named, list) and all(isinstance(f, str) for f in named)):
        r.error("output_contract.host_values must be a list of field names")
    else:
        if named and constrained is not True:
            r.error("output_contract.host_values needs a constrained answer")
        for f in named:
            if not (f == "label" if answer_type == "label" else
                    answer_type == "json" and schema_field(schema, f) is not None):
                r.error(f"output_contract.host_values names {f!r}, which is not a field of this answer")
    temp = m.get("temperature", 0)
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
        r.error("temperature must be a number from 0 to 2")
        temp = 0
    elif otype in ("label", "json", "reasoned") and temp > 0:
        r.warn(f"temperature {temp} is ignored: a {otype} answer runs at 0", "temperature_ignored")
        temp = 0
    r.temperature = temp
    hist = m.get("history")
    if hist is not None:
        shape = hist.get("shape") if isinstance(hist, dict) else None
        if shape not in HISTORY_SHAPES:
            r.error(f"history.shape must be one of {', '.join(HISTORY_SHAPES)}")
        elif shape == "as_sent":
            if not (_whole(hist.get("max_turns")) and hist["max_turns"] >= 1):
                r.error("history.max_turns must be a whole number, at least 1")
            if otype != "text" or kind == "grounded":
                r.error("history 'as_sent' needs a text answer and a kind other than grounded")
    pc = m.get("prompt_check")
    if pc is None:
        r.warn("no prompt_check: nothing proves the prompt matches training (§8.6)", "prompt_unchecked")
    elif not (isinstance(pc, dict) and isinstance(pc.get("canonical"), str) and "input" in pc):
        r.error("prompt_check must be an object with 'input' and a string 'canonical'")
    skill = m.get("skill")
    if skill:
        name = skill.get("name") if isinstance(skill, dict) else None
        version = _version_of(skill.get("version")) if isinstance(skill, dict) else None
        if not isinstance(name, str) or not name or version is None:
            r.error("pigeon.json 'skill' needs a 'name' and a whole-number 'version'")
    # The card is display text only (D8): one that breaks §11 is skipped with a
    # warning naming the field, and the Pigeon still loads.
    if "card" in m:
        from .manifest import card_problems
        problems = card_problems(m.get("card"))
        if problems:
            r.warn("card skipped: " + "; ".join(problems), "card_skipped")
    if not r.ok:
        raise _Stop

    # 5. Files by kind.
    if base_only and "adapter.lora.gguf" in by_name:
        r.error("a base Pigeon (base_only) carries no adapter.lora.gguf")
        raise _Stop
    need_files = [*([] if base_only else ["adapter.lora.gguf"]), *REQUIRED_FILES.get(kind, ())]
    if skill:
        need_files.append("skill/skill.json")
    mat = m.get("material") if isinstance(m.get("material"), dict) else {}
    method = mat.get("retrieval")
    from . import embed
    if kind == "grounded" and method == "hybrid-v1":
        need_files.append(embed.EMBEDDINGS_FILE)
    # §8.6 (D38): the picture prompt_check names must be in the file, or no
    # Runner can rebuild the prompt it proves.
    pc_input = pc.get("input") if isinstance(pc, dict) else None
    if isinstance(pc_input, dict) and isinstance(pc_input.get("image"), str):
        need_files.append(pc_input["image"])
    for f in need_files:
        if f not in by_name:
            if f == "adapter.lora.gguf":
                r.error("adapter.lora.gguf is missing — there is nothing to load", "missing_required_file")
            elif f == "skill/skill.json":
                r.error("skill/skill.json is missing — a recipe cannot be rendered "
                        "without the parts it names", "missing_required_file")
            elif f == embed.EMBEDDINGS_FILE:
                r.error(f"{embed.EMBEDDINGS_FILE} is missing — hybrid-v1 has nothing to search with",
                        "missing_required_file")
            else:
                r.error(f"{f} is required for kind '{kind}' and is missing", "missing_required_file")
            raise _Stop
    try:
        lora = base_only or read_header_bytes(by_name["adapter.lora.gguf"]).get("adapter.type") == "lora"
    except NotGGUF:
        lora = False
    if not lora:
        r.error("adapter.lora.gguf is not a GGUF LoRA adapter (its header must say adapter.type = lora)",
                "not_a_lora")
        raise _Stop
    if SHOTS_FILE in by_name:
        problem = shots_problem(by_name[SHOTS_FILE])
        if problem:
            r.error(problem, "bad_payload")
            raise _Stop

    # The material slot and the grounded kind imply each other, in both
    # directions: a grounded Pigeon with no slot answers from nothing, and a slot
    # on any other kind reaches a reader that sends the literal text "{material}".
    if kind == "grounded":
        if MATERIAL_SLOT not in template:
            r.error(f"kind 'grounded' needs {MATERIAL_SLOT} in prompt_template — "
                    "that slot is where the app pastes the material", "material_slot_missing")
            raise _Stop
        if not isinstance(m.get("material"), dict):
            r.error("pigeon.json is missing 'material'", "missing_field")
            raise _Stop
        if method not in RETRIEVALS:
            # Named, not guessed: answering from the wrong sections looks exactly
            # like answering from the right ones.
            r.error(f"material.retrieval {method!r} is not one this version can do "
                    f"({', '.join(RETRIEVALS)}) — update the app", "unknown_retrieval")
            raise _Stop
        if not _whole(mat.get("sections")) or mat["sections"] < 1:
            r.error("material.sections must say how many sections the material "
                    "was chopped into, and it must be at least one", "no_sections")
            raise _Stop
        for problem in _chunks_problems(by_name["material/chunks.json"]):
            r.error(problem, "bad_payload")
            raise _Stop
        if method == "hybrid-v1":
            # The embedding model is a second pinned identity, and here the hash
            # is the rule (§9.4).
            ordinals = [c.get("ordinal") for c in json.loads(by_name["material/chunks.json"])["chunks"]]
            for problem in embed.embeddings_problems(by_name[embed.EMBEDDINGS_FILE], ordinals, model_sha=None):
                r.error(problem, "bad_payload")
                raise _Stop
            emb = json.loads(by_name[embed.EMBEDDINGS_FILE])
            digest = embed.chunks_digest(json.loads(by_name["material/chunks.json"]))
            if emb.get("chunks_sha256") != digest:
                r.error("material/embeddings.json was made from different sections (chunks_sha256)",
                        "bad_payload")
                raise _Stop
            held = [embed.MODEL["sha256"]] if embedding_models is None else embedding_models
            model = emb.get("model") or {}
            if held is not False and model.get("sha256") not in held:
                r.error(f"material/embeddings.json was made with {model.get('file')}, an embedding model "
                        f"this app does not hold (it carries {embed.MODEL['file']}) — update the app",
                        "embedding_model_mismatch")
                raise _Stop
    elif MATERIAL_SLOT in template:
        r.error(f"prompt_template has {MATERIAL_SLOT} but kind is {kind!r} — "
                "only a grounded Pigeon carries material to put there", "material_slot_on_wrong_kind")
        raise _Stop

    if skill:
        # Accepted by how it is drawn, not by its name (§10.3). A method this app
        # cannot perform is refused by name: the wrong method draws a well-formed
        # wrong picture.
        for problem in _skill_payload_problems(by_name["skill/skill.json"], skill["name"],
                                               _version_of(skill["version"])):
            r.error(problem, "unknown_render" if "cannot do" in problem else "bad_payload")
            raise _Stop

    pictures = "image" in inputs
    if "actions.json" in by_name:
        # D54: display-and-pressing data only, so like a bad card it is skipped,
        # never a refusal — and then nothing is pressed.
        from .runner import actions_problem, usable_actions
        problem = actions_problem(by_name["actions.json"], m)
        r.actions = usable_actions(by_name["actions.json"], m)
        if problem:
            r.warn(f"actions.json skipped, so this Pigeon is not played: {problem}", "actions_skipped")
    tf = m.get("trained_from")
    if pictures and isinstance(tf, dict) and tf.get("precision") == "4bit":
        r.warn("trained on a 4-bit base: a picture Pigeon trained this way answers measurably "
               "worse when served than its training scores suggest", "trained_quantized")
    if pictures:
        mm = bm.get("mmproj")
        if not isinstance(mm, dict) or not isinstance(mm.get("sha256"), str) or not mm["sha256"]:
            r.error("a picture Pigeon must name its projector by hash (base_model.mmproj.sha256)",
                    "projector_unnamed")
            raise _Stop
        if "projector.gguf" in by_name and hashlib.sha256(by_name["projector.gguf"]).hexdigest() != mm["sha256"]:
            r.error("projector.gguf does not hash to base_model.mmproj.sha256", "projector_mismatch")
            raise _Stop
        if "projector.gguf" in by_name:
            # D57: the right bytes by hash can still be no projector at all.
            try:
                read_header_bytes(by_name["projector.gguf"])
            except (NotGGUF, ValueError, IndexError, UnicodeDecodeError):
                r.error("projector.gguf is not a GGUF file", "bad_payload")
                raise _Stop

    r.self_test = self_test_for(m)
    r.examples = sum(1 for f in r.files if f.startswith("examples/") and f.endswith(".expected.txt"))

    # 6. The base — identity refuses, the hash is a fast path, another file warns.
    if bases is None:
        return
    want = {"arch": bm.get("arch"), "name": bm.get("name"), "size": bm.get("size"),
            "finetune": bm.get("finetune")}
    r.base_needed = bm.get("file")
    named = f"{bm.get('file')} ({want['arch']}, {want['name']} {want['size']}" + \
            (f", {want['finetune']})" if want["finetune"] else ")")
    from . import registry as registry_mod
    entry = registry_mod.entry_for(registry, want) if registry else None
    if entry and registry_mod.disagrees(entry, bases, lambda b: hash_check(b["path"])):
        r.warn(f"the registry lists a file as {entry.get('id')} whose own header is a different model; "
               f"that entry is ignored", "registry_disagrees")
        entry = None
    if not bases:
        r.base_match = BASE_MISSING
        r.needs = registry_mod.needs(want, bm.get("file"), entry)
        r.error(f"needs the base it was trained on — {named} — and this machine has no models. "
                f"It will not load onto a different base; download that model first", "no_base")
        raise _Stop
    exact = [b for b in bases if b.get("file") == bm.get("file")]
    if exact and bm.get("sha256") and any(hash_check(b["path"]) == bm["sha256"] for b in exact):
        r.base_match = BASE_EXACT
    elif any(same_base(want, b) or (entry and registry_mod.header_is_entry(entry, b)) for b in bases):
        r.base_match = BASE_SAME_OTHER_QUANT
        have = ", ".join(sorted({b["file"] for b in bases
                                 if same_base(want, b) or (entry and registry_mod.header_is_entry(entry, b))}))
        r.warn(f"trained against {bm.get('file')}; this machine has the same base as {have}. "
               f"It loads, but that is not the file it was measured on", "same_base_other_quant")
    else:
        r.base_match = BASE_WRONG
        r.needs = registry_mod.needs(want, bm.get("file"), entry)
        r.error(f"needs the base it was trained on — {named} — and this machine does not have it. "
                f"It will not load onto a different base; download that model first", "wrong_base")
        raise _Stop
    if pictures and "projector.gguf" not in by_name and projectors is not None \
            and bm["mmproj"]["sha256"] not in projectors:
        if bm["mmproj"]["sha256"] in registry_mod.projector_hashes(entry):
            return          # downloadable: the registry lists it for this base
        r.needs = registry_mod.needs(want, bm["mmproj"].get("file"), entry, "projectors")
        r.error(f"needs its projector {bm['mmproj'].get('file')} and this machine does not have it",
                "no_projector")
        raise _Stop
