"""pigeon.json, written from the run's own config — never retyped.

Spec §3.2. The field that matters most is `prompt_template`: the first
trained pilot Pigeon looked broken on its check ride purely because it was asked a
bare question in a shape it had never seen. The shape an adapter was trained
against has to travel with it, because an adapter is not source and cannot be
refactored.

Everything here is read off the run folder Flight School wrote: base.json
for the base, plan.json for the job, measured.json for the answer length,
check-ride.json for the score, and the base GGUF's own header for its
identity. A field that cannot be read is an error, not a default — a manifest
with a guessed base is worse than no manifest.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path

from .gguf import base_identity, quant_name, read_header, sha256_file

FORMAT_VERSION = 1
KINDS = ("classifier", "extractor", "persona", "router", "freeform", "grounded", "skillpack")

# A `skillpack` is the one kind that ships CODE (under `code/`). It carries no
# adapter and no prompt template, and it MUST carry a seal by a key the app
# trusts (seal.py, docs/PIGEON-FORMAT.md "Seal"). A `code/` entry in any other
# kind is refused, sealed or not.
CODE_KINDS = ("skillpack",)

# A `grounded` Pigeon answers from material handed to it at ask time, never
# from what the weights remember. The material travels inside the file and the
# APP pastes it in — a Pigeon cannot go looking for anything.
#
# The slot is a named part of the template rather than a convention, because
# the position the material occupies at ask time must be the position it
# occupied at training time. Measured: an adapter trained on (passage,
# question) reads the passage it is given and prefers it over what it
# memorised (11/14 on a fact swap, 0 recitations — the baseline notes, "Step 0 of the
# document Pigeon"). A reader that pasted the material somewhere else, or not
# at all, would get fluent answers that quietly ignore it.
MATERIAL_SLOT = "{material}"

# A Pigeon whose answer is a RECIPE rather than the finished thing: the model
# names parts from a closed vocabulary and a SKILL turns them into the artefact.
# The skill travels in the file as data (`skill/skill.json`), and a reader
# accepts it by HOW IT IS DRAWN, not by what it is called (§9.3). The parts
# table ships inside the file and its name and version must still equal
# pigeon.json's, so a list of skill names in every app protected nothing more —
# and it meant a new skill opened nowhere until every app was updated.
#
# Measured (the baseline notes, "T2"): asked for a drawing directly, an adapter produces
# perfectly styled output that matches nothing — 0/20 on every part at 137
# designs — because every coordinate is a knife-edge token. Asked for the recipe,
# the same corpus carries real information. The skill is what makes that answer
# useful to anyone else.
#
# A renderer is a method and the versions of it this app performs. `concat` v1
# is §9.2's: `head`, then the fragment for each named part in `order`, inside
# `wrapper`. A method this app cannot perform is refused by name, exactly like an
# unknown `kind`: drawing with the wrong method makes a well-formed wrong picture.
# Pigeon's reader holds the same table (`skill.rs::RENDERERS`).
RENDERERS: dict = {"concat": (1,)}
# Skills packed before `render` existed. Each was drawn by concatenation, so each
# is read as `concat` v1; every other skill must say how it is drawn.
LEGACY_SKILLS = {("face-grammar", 1)}


def version_number(v):
    """A version written as `1` or `"1"`; anything else is no version."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


SHOTS_FILE = "shots.json"


def shots_problem(raw: bytes) -> str | None:
    """D87: `shots.json` is a list of `{"user", "pigeon"}` turns, both text."""
    try:
        shots = json.loads(raw)
    except ValueError:
        return "shots.json is not JSON"
    if not isinstance(shots, list) or not all(
            isinstance(t, dict) and isinstance(t.get("user"), str) and isinstance(t.get("pigeon"), str)
            for t in shots):
        return "shots.json must be a list of turns, each with a user and a pigeon, both text"
    return None


def schema_field(schema, path: str):
    """D88: the node of `output_schema` a `host_values` name points at, or None.
    Names joined by `.` step into `properties`; `name[]` then steps into the
    array's `items`, so `steps[].skill` is the `skill` of every step."""
    node = schema
    for seg in path.split("."):
        items = seg.endswith("[]")
        name = seg[:-2] if items else seg
        props = node.get("properties") if isinstance(node, dict) else None
        if not name or not isinstance(props, dict) or not isinstance(props.get(name), dict):
            return None
        node = props[name]
        if items:
            node = node.get("items")
            if not isinstance(node, dict):
                return None
    return node


def with_values(schema, values: dict):
    """D88: the schema with each named field held to the host's values, an
    `enum` of exactly those in the order given (pigeon-core values.rs). None
    when a name is not a field of the schema."""
    out = json.loads(json.dumps(schema))
    for path, allowed in (values or {}).items():
        node = schema_field(out, path)
        if node is None:
            return None
        node.pop("pattern", None)
        node.pop("const", None)
        node["enum"] = list(allowed)
    return out


def renderer_list() -> str:
    return ", ".join(f"{m} v{v}" for m, vs in RENDERERS.items() for v in vs)


def skill_payload_problems(p, name, version) -> list:
    """`skill/skill.json` checked the way Pigeon parses it (`skill.rs::Payload`).

    In order: it is the skill the manifest names, at its version; it has an
    `order`; it is drawn by a method this app performs (or is a legacy skill,
    read as concatenation); and it has concatenation's shape — one `{parts}`
    slot, and a table of strings for every key in `order`. `order` is required
    because a JSON object has no order a second reader is obliged to keep, so
    two apps could otherwise draw one recipe two ways."""
    if not isinstance(p, dict):
        return ["skill/skill.json must be an object"]
    if not isinstance(p.get("name"), str) or "version" not in p:
        return ["skill/skill.json needs a 'name' and a 'version'"]
    if p["name"] != name or version_number(p["version"]) != version:
        return [f"skill/skill.json is {p['name']!r} v{version_number(p['version'])}, but pigeon.json "
                f"names {name!r} v{version} — a recipe is only drawn with the table it was trained against"]
    order = p.get("order")
    if not isinstance(order, list) or not all(isinstance(k, str) for k in order):
        return ["skill/skill.json needs 'order', a list of part keys"]
    if not order:
        return ["skill/skill.json has no `order`, so the parts cannot be drawn the same way twice"]

    render = p.get("render")
    if render is None:
        if (name, version) not in LEGACY_SKILLS:
            return [f"skill/skill.json does not say how {name!r} is drawn — it needs "
                    '"render": {"method": "concat", "version": 1}']
        method, method_version = "concat", 1
    elif not isinstance(render, dict) or not isinstance(render.get("method"), str):
        return ["skill/skill.json 'render' must be an object with a 'method' and a 'version'"]
    else:
        method, method_version = render["method"], version_number(render.get("version"))
    if method_version not in RENDERERS.get(method, ()):
        return [f"this Pigeon's skill is drawn with {method!r} v{method_version}, which this app "
                f"cannot do (it can draw with {renderer_list()}) — update the app"]

    for key in ("wrapper", "head"):
        if key in p and not isinstance(p[key], str):
            return [f"skill/skill.json '{key}' must be a string"]
    if p.get("wrapper", "{parts}").count("{parts}") != 1:
        return ["skill/skill.json 'wrapper' must contain {parts} exactly once"]
    parts = p.get("parts")
    if not isinstance(parts, dict):
        return ["skill/skill.json needs a 'parts' object"]
    for key in order:
        table = parts.get(key)
        if not isinstance(table, dict):
            return [f"skill/skill.json has no parts table for {key!r}"]
        if not all(isinstance(f, str) for f in table.values()):
            return [f"skill/skill.json parts.{key} must map each name to a string"]
    return []

# How the app finds what to paste. `whole` means paste all of it, every time —
# the honest choice when it fits, with nothing to get wrong. `keyword-overlap-v1`
# is chunker.rs's deterministic search: word overlap against each section, with
# a floor below which the honest answer is "no section," so the model is told
# the material does not cover it rather than being grounded in noise.
#
# A reader that does not know a named method MUST refuse the file rather than
# guess, which is why the name carries a version.
RETRIEVALS = ("whole", "keyword-overlap-v1", "keyword-overlap-v2", "hybrid-v1")

# Under this many estimated tokens the material is pasted whole. Chosen to sit
# well inside the smallest base's window alongside a question and an answer;
# it is a budget, not a measurement of any one model.
WHOLE_MATERIAL_CEILING = 3000

_LABEL_LIST = re.compile(r"(?:one of|exactly one of)\s*:?\s*(.+)", re.I)


def label_set(plan: dict) -> list[str] | None:
    """The same rule fly.py grades by: an `output_shape` that names a fixed
    label list makes this a classifier, and exact match is the honest test."""
    m = _LABEL_LIST.search(plan.get("output_shape", "") or "")
    if not m:
        return None
    parts = [p.strip(" .").strip() for p in re.split(r",| or ", m.group(1))]
    labels = [p for p in parts if p and len(p) < 40]
    return labels if len(labels) >= 2 else None


def est_tokens(text: str) -> int:
    """Deliberately chunker.rs's ratio (chars/3.5), so the Rust side that
    chopped the material and the Python side that packs it agree about how big
    it is. An estimate, and it gates a choice of method, never training."""
    return math.ceil(len(text) / 3.5)


def material_dir(run_dir: Path | str) -> Path | None:
    """The folder holding this run's reference material, or None.

    Two names are accepted: `material/`, which the packer writes, and `refs/`,
    which is where the lesson flow's chunker already puts a copy of every
    reference file beside its chunks.json. Reading the lesson layout directly
    means a grounded Pigeon can be packed from a lesson that already ran.
    """
    for name in ("material", "refs"):
        d = Path(run_dir) / name
        if (d / "chunks.json").exists():
            return d
    return None


def read_material(d: Path) -> dict:
    """The manifest's `material` block, measured off the folder. With
    `embeddings.json` beside the chunks — made at pack time by embed.py — the
    method is `hybrid-v1` and the block names the embedding model, a second
    pinned identity beside the base."""
    index = json.loads((d / "chunks.json").read_text())
    chunks = index.get("chunks") or []
    text = "".join(str(c.get("text", "")) for c in chunks)
    files = sorted({str(c.get("file")) for c in chunks if c.get("file")})
    total = est_tokens(text)
    out = {
        "retrieval": "whole" if total <= WHOLE_MATERIAL_CEILING else "keyword-overlap-v2",
        "sections": len(chunks),
        "files": files,
        "est_tokens": total,
    }
    emb = d / "embeddings.json"
    if emb.is_file():
        from . import embed
        try:
            e = json.loads(emb.read_text())
        except ValueError:
            e = None
        if isinstance(e, dict) and (e.get("model") or {}).get("sha256") == embed.MODEL["sha256"] \
                and e.get("chunks_sha256") == embed.chunks_digest(index) and e.get("units"):
            m = e["model"]
            out["retrieval"] = embed.RETRIEVAL
            out["embedding"] = {"model": m["name"], "file": m["file"], "sha256": m["sha256"],
                                "dims": int(m["dims"]), "units": len(e["units"])}
    return out


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "pigeon"


def _read(run_dir: Path, name: str, required: bool = True):
    p = run_dir / name
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{name} is missing from the run folder — the manifest cannot be written from a guess")
        return None
    return json.loads(p.read_text())


def prompt_template_for(vision: bool, grounded: bool) -> str:
    """The user turn a Flight School run trains with — also what its check ride
    fills (instructor/fly.py), so the two cannot drift apart."""
    if vision and grounded:
        return "{image}\n{material}\n\n{input}"
    if vision:
        return "{image}\n{input}"
    if grounded:
        return "{material}\n\n{input}"
    return "{input}"


def answer_cap(plan: dict, measured_tokens: int, *, grounded: bool) -> int:
    """`output_contract.max_tokens`: what was MEASURED, never what the plan
    claimed (the poet plan claimed 120 tokens against 303 real ones and starved
    its own test), with room above it. A Runner MUST stop here (§5.4, §8.5), so
    the check ride stops here too: a ride allowed further would record answers
    no Runner can give (found by the engine on pat-v2: 172 in the file, more in
    the ride)."""
    claimed = int(plan.get("max_answer_tokens") or 40)
    reasoned = (plan.get("grader") or {}).get("rule") == "instructor.rules_table:reasoned"
    if label_set(plan) and not reasoned and not grounded:
        return max(8, min(16, measured_tokens + 4)) if measured_tokens else 12
    return max(16, int(max(measured_tokens, claimed) * 1.5))


def detect_kind(plan: dict, state: dict, material: dict | None = None) -> str:
    """What this adapter is, from what it was built as.

    A fixed label set is a classifier whatever the modality. Everything else
    Quick Build makes today is `freeform` — including a voice. A voice trained
    here is NOT a `persona` in the spec's sense: persona requires a
    system_prompt.txt that the runtime injects, and Quick Build strips the
    system prompt before training so the adapter works with none. Shipping a
    prompt the adapter never saw, to be injected on every call, would make it
    worse, not better. `persona` is reserved for runs that genuinely pair an
    adapter with a prompt.
    """
    # Grounded wins over a label set: what decides the kind here is the SHAPE
    # the adapter is asked in, and a grounded job's shape has a second slot in
    # it. A grounded job with a fixed label list would want both the labels and
    # the slot; nothing produces one, and it is listed as open in §8 of
    # docs/PIGEON-FORMAT.md rather than guessed at here.
    if material:
        return "grounded"
    if label_set(plan):
        return "classifier"
    return "freeform"


def _from_source(src: dict) -> tuple[dict, dict, dict, dict]:
    """A hand-graded lesson is not a Quick Build run: it has no plan.json,
    no measured.json and no scored check ride. `export/pigeon-source.json`
    (written by lesson.rs) says the same things in one place — the adapter,
    the base, what the job is, the approved pairs — and this shapes it into
    the four documents a run folder would carry. Nothing is guessed: the
    answer budget is measured over the approved answers, the example count
    is the approved count, and there is no flight, so no report."""
    answers = [str(e.get("answer", "")) for e in src.get("examples", [])]
    longest = max((len(a) for a in answers), default=0)
    measured_tokens = int(longest / 3.5) + 1 if longest else 0
    plan = {
        "job_name": src.get("name") or "lesson",
        "one_line": src.get("one_line") or "",
        "input_shape": src.get("input_shape") or "one message",
        "output_shape": src.get("output_shape") or "one answer, in the shape the lesson taught",
        "modality": "text",
        "max_answer_tokens": measured_tokens or 40,
    }
    state = {
        "stage": "complete",
        "adapter": src.get("adapter"),
        "creative": False,
        "dataset": {"train": int(src.get("example_count") or len(answers)), "valid": 0, "holdout": 0},
        "checked_by": src.get("checked_by") or "asking",
    }
    base = {"student_gguf": src["base_gguf"], "student_repo": src.get("base_repo")}
    measured = {"answer_tokens_measured": measured_tokens, "answer_tokens_claimed": measured_tokens}
    return plan, state, base, measured


# ---------------------------------------------------------------------------
# the card — what a reader shows before anyone has used the Pigeon
#
# Optional (PIGEON-FORMAT §2.1). Every field is the MAKER'S claim or a count
# read off the run, and a reader shows it as such. Text is refused when it is
# too long, never cut: a cut sentence is a different sentence.

CARD_TEXT_LIMITS = {"does": 60, "input_hint": 40, "output_noun": 24, "talks_like": 60}
CARD_MAX_EXAMPLES = 5


def card_problems(card) -> list[str]:
    """The card checked the way a reader parses it. Unknown keys are allowed —
    a later version may add one, and an older reader skips what it does not
    know rather than refusing the Pigeon."""
    if not isinstance(card, dict):   # null included: a card that is there must be an object (D58)
        return ["card must be an object"]
    out = []
    does = card.get("does")
    if not isinstance(does, str) or not does.strip():
        out.append("card.does is required — one line saying what the Pigeon does")
    for key, limit in CARD_TEXT_LIMITS.items():
        v = card.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            out.append(f"card.{key} must be text")
        elif len(v) > limit:
            out.append(f"card.{key} is {len(v)} characters; it may be at most {limit} — shorten it, it is never cut")
    ex = card.get("examples")
    if ex is not None:
        if not isinstance(ex, list) or len(ex) > CARD_MAX_EXAMPLES:
            out.append(f"card.examples must be a list of at most {CARD_MAX_EXAMPLES}")
        else:
            for i, e in enumerate(ex):
                if not isinstance(e, dict) or not isinstance(e.get("input"), str) or not isinstance(e.get("output"), str):
                    out.append(f"card.examples[{i}] needs text 'input' and 'output'")
    stats = card.get("input_stats")
    if stats is not None:
        def num(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0
        ok = isinstance(stats, dict) and all(
            isinstance(stats.get(k), dict) and num(stats[k].get("p50")) and num(stats[k].get("p95"))
            for k in ("words", "lines"))
        if not ok:
            out.append("card.input_stats needs 'words' and 'lines', each with numbers 'p50' and 'p95'")
    check = card.get("check")
    if check is not None:
        whole = lambda x: isinstance(x, int) and not isinstance(x, bool)  # noqa: E731
        if not (isinstance(check, dict) and whole(check.get("passed")) and whole(check.get("total"))
                and 0 <= check["passed"] <= check["total"]):
            out.append("card.check needs whole numbers 'passed' and 'total', with passed ≤ total")
    rev = card.get("revision")
    if rev is not None and (not isinstance(rev, int) or isinstance(rev, bool) or rev < 1):
        out.append("card.revision must be a whole number from 1")
    return out


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]


def build_card(*, plan: dict, kind: str, flight: dict | None, source: dict | None, pool: list[dict],
               does: str | None = None, input_hint: str | None = None, output_noun: str | None = None,
               talks_like: str | None = None, revision: int = 1) -> tuple[dict, list[str]]:
    """The card for a run or a lesson, and its problems. Examples come from the
    check ride's PASSED rows (held out — never the practice set, never the
    control) for a run, and from the approved pairs for a lesson."""
    from .pack import pick_examples  # pack imports this module
    card: dict = {"does": (does if does is not None else (plan.get("one_line") or "")).strip()}
    if input_hint is not None:
        if input_hint.strip():
            card["input_hint"] = input_hint.strip()
    elif plan.get("input_shape") and len(plan["input_shape"]) <= CARD_TEXT_LIMITS["input_hint"]:
        # A default is only taken when it fits; the plan's words are not cut.
        card["input_hint"] = plan["input_shape"]
    noun = output_noun if output_noun is not None else ("label" if kind == "classifier" else None)
    if noun and noun.strip():
        card["output_noun"] = noun.strip()
    if kind == "persona" and talks_like and talks_like.strip():
        card["talks_like"] = talks_like.strip()

    if source:
        rows = [{"question": str(e.get("question", "")), "answer": str(e.get("answer", "")),
                 "group": e.get("group") or e.get("category") or ""} for e in source.get("examples", [])]
    else:
        groups = {r.get("question"): r.get("group") or "" for r in pool}
        rows = [{"question": r["question"], "answer": str(r["expected"]), "group": groups.get(r["question"], "")}
                for r in (flight or {}).get("rows", [])
                if r.get("passed") is True and isinstance(r.get("question"), str)
                and r.get("expected") is not None and not r.get("image")]
    picked = pick_examples(rows, n=CARD_MAX_EXAMPLES, seed=11) if rows else []
    if picked:
        card["examples"] = [{"input": e["question"], "output": e["answer"], **({"group": e["group"]} if e.get("group") else {})}
                            for e in picked]

    inputs = [r["question"] for r in pool if isinstance(r.get("question"), str) and r["question"]]
    if inputs:
        words = [len(t.split()) for t in inputs]
        lines = [t.count("\n") + 1 for t in inputs]
        card["input_stats"] = {"words": {"p50": _percentile(words, 0.5), "p95": _percentile(words, 0.95)},
                               "lines": {"p50": _percentile(lines, 0.5), "p95": _percentile(lines, 0.95)}}
    if flight and isinstance(flight.get("passed"), int) and isinstance(flight.get("total"), int):
        card["check"] = {"passed": flight["passed"], "total": flight["total"]}
    card["revision"] = int(revision)
    return card, card_problems(card)


def build_manifest(run_dir: Path | str, *, name: str | None = None,
                   author: str | None = None, trained_with: str = "Flight School",
                   check_hash: bool = True, does: str | None = None, input_hint: str | None = None,
                   output_noun: str | None = None, talks_like: str | None = None,
                   revision: int = 1) -> tuple[dict, dict]:
    """Returns (manifest, facts). `facts` carries what the packer needs that
    is not itself in the manifest — the label list, the modality, the base's
    path — so the two are read from the run exactly once.

    Two folder shapes are read: a Quick Build run (state.json, plan.json,
    base.json, measured.json) or a lesson with `export/pigeon-source.json`."""
    run_dir = Path(run_dir)
    source = _read(run_dir / "export", "pigeon-source.json", required=False)
    if source:
        plan, state, base, measured = _from_source(source)
        flight = None
    else:
        state = _read(run_dir, "state.json")
        plan = state.get("plan") or _read(run_dir, "plan.json")
        base = _read(run_dir, "base.json")
        measured = _read(run_dir, "measured.json", required=False) or {}
        flight = state.get("flight")

    base_path = Path(base["student_gguf"])
    if not base_path.exists():
        raise FileNotFoundError(f"the base this run trained against is not on disk: {base_path}")
    header = read_header(base_path)
    ident = base_identity(header)
    for key, field in (("arch", "general.architecture"), ("name", "general.basename or general.name"),
                       ("size", "general.size_label")):
        # An empty field matches nothing in any reader (spec §7.2), so a Pigeon
        # naming its base this way could only ever load on the exact file.
        if not ident[key]:
            raise ValueError(f"{base_path.name} carries no {field} — it cannot be named as a base")

    vision = plan.get("modality") == "vision"
    # A reasoning table answers <think>…</think><do>LABEL</do> (Flight School's
    # "Reasoning, then a function"); spec §5.4 / D26 calls that `reasoned`. Its
    # plan states no label list, so the labels are the table's own labels.txt.
    reasoned = (plan.get("grader") or {}).get("rule") == "instructor.rules_table:reasoned"
    labels = label_set(plan)
    if reasoned and not labels:
        lt = run_dir / "labels.txt"
        labels = [l.strip() for l in lt.read_text().splitlines() if l.strip()] if lt.is_file() else None
        if not labels:
            raise ValueError("a reasoning Pigeon needs its labels (labels.txt): the answer inside "
                             "<do>…</do> is checked against them")
    skill = (state.get("skill") or plan.get("skill") or None)
    if skill:
        # The same check an importer runs, before the file exists: a run whose
        # skill another app would refuse is not packed at all.
        problems = skill_payload_problems(skill.get("payload"), skill.get("name"),
                                          version_number(skill.get("version", 1)))
        if problems:
            raise ValueError("the run's skill cannot be packed: " + "; ".join(problems))
    mat_dir = material_dir(run_dir)
    material = read_material(mat_dir) if mat_dir else None
    # A Reader (Flight School's kinds screen) trains every row on ONE section
    # above the question, and an adapter is only correct in the shape it was
    # trained in — so its material is searched section by section at ask
    # time whatever its size, never pasted whole. `whole` stays for a run
    # that trained that way.
    if material and plan.get("kind") == "reader" and material["retrieval"] == "whole":
        material["retrieval"] = "keyword-overlap-v2"
    kind = "classifier" if reasoned and not material else detect_kind(plan, state, material)

    # The exact wrapper the check ride used. The training data carried NO
    # system prompt (the brief is stripped before training — that is the
    # distillation trick the app is built on), and the ride asks one user
    # turn. So the template is the user content, rendered through the base's
    # own chat template. For vision the image precedes the text.
    # A grounded run adds the material slot ABOVE the question, which is where
    # the training rows put it. Byte for byte: the app reproduces this template
    # and fills both slots.
    prompt_template = prompt_template_for(vision, bool(material))

    # Answer budget: the same number the check ride stopped at (answer_cap).
    max_tokens = answer_cap(plan, int(measured.get("answer_tokens_measured") or 0), grounded=bool(material))

    ds = state.get("dataset") or {}
    example_count = int(ds.get("train", 0)) + int(ds.get("valid", 0))

    manifest = {
        "format_version": FORMAT_VERSION,
        "name": name or plan.get("job_name") or run_dir.name,
        "id": slugify(name or plan.get("job_name") or run_dir.name),
        "kind": kind,
        "base_model": {
            "file": base_path.name,
            "sha256": sha256_file(base_path) if check_hash else None,
            "arch": ident["arch"],
            "name": ident["name"],
            "size": ident["size"],
            "repo": ident["repo"],
            "quant": quant_name(header),
            "vision": vision,
        },
        # The temperature the check ride asked at (§5.5). Flight School always
        # asks at 0, and label, JSON and reasoned answers are locked there.
        "temperature": (flight or {}).get("temperature", 0),
        "prompt_template": prompt_template,
        "output_contract": output_contract(kind, skill, reasoned, max_tokens, flight),
        "inputs": ["image", "text"] if vision else ["text"],
        "trained_on": date.today().isoformat(),
        "trained_with": trained_with,
        "example_count": example_count,
        "author": author,
        "license": "Apache-2.0",
        # Reserved, empty in v1. The Airfield already decided the policy: a
        # signature proves who packed a thing, never that it is good.
        "signature": "",
    }
    if ident.get("finetune"):
        # §7.2 rule 4 (D12): written whenever the header has it, so an Instruct
        # Pigeon never loads on the Thinking model of the same size.
        manifest["base_model"]["finetune"] = ident["finetune"]
    tf = trained_from(base.get("student_repo"), _hf_home(run_dir))
    if tf:
        # Top level, as §5.1's example and the reference reader place it.
        manifest["trained_from"] = tf
    if skill:
        manifest["skill"] = {"name": skill["name"], "version": int(skill.get("version", 1))}
    if material:
        manifest["material"] = material
    projector_path = None
    if vision:
        # Spec §3.7, open item, leaned into: the projector is a second file
        # and gets its own hash rather than being implied from the base.
        # Since 2026-09-21 the file also CARRIES it (projector.gguf), so the
        # hash is always taken: it is what an importer checks the carried
        # bytes against, and a projector nobody can check is not shipped.
        entry, projector_path = _mmproj_entry(run_dir, True)
        manifest["base_model"]["mmproj"] = entry

    facts = {
        "labels": labels, "vision": vision, "base_path": str(base_path),
        # Where the packer reads the material from, so it is found once.
        "material_dir": str(mat_dir) if mat_dir else None, "material": material,
        "projector": str(projector_path) if projector_path else None,
        "skill": skill,
        "plan": plan, "state": state, "flight": flight, "measured": measured,
        # For a lesson: the approved pairs the examples are drawn from, and
        # the fact that it was checked by asking, not scored.
        "source": source,
    }
    from .pack import example_pool  # pack imports this module
    card, issues = build_card(plan=plan, kind=kind, flight=flight, source=source,
                              pool=example_pool(run_dir, facts), does=does, input_hint=input_hint,
                              output_noun=output_noun, talks_like=talks_like, revision=revision)
    # Words someone TYPED for the card that do not fit are refused: the draft
    # shows why and packing stops, rather than shipping a cut sentence. A
    # default from the plan that does not fit is not anyone's mistake, so the
    # file simply goes without a card and the draft says so.
    typed = any(v is not None for v in (does, input_hint, output_noun, talks_like))
    note = None
    if issues and not typed:
        note = "no card: " + "; ".join(issues) + " — write a shorter one on the Package step to include a card"
        issues = []
        card = None
    if card and not issues:
        manifest["card"] = card
    facts["card"], facts["card_problems"], facts["card_note"] = card, issues, note
    return manifest, facts


def output_contract(kind: str, skill, reasoned: bool, max_tokens: int, flight) -> dict:
    """§5.4. `constrained` is written true only when the check ride decoded that
    way (instructor/fly.py), because the maker's test and every Runner must."""
    if skill:
        # A recipe is JSON with fixed keys: naming them lets a reader check the
        # shape without knowing the skill.
        return {"type": "json", "fields": list(skill.get("fields") or []), "max_tokens": max_tokens}
    if reasoned:
        return {"type": "reasoned", "answer": "label", "max_tokens": max_tokens,
                "constrained": bool((flight or {}).get("constrained"))}
    if kind == "classifier":
        return {"type": "label", "max_tokens": max_tokens,
                "constrained": bool((flight or {}).get("constrained"))}
    return {"type": "text", "max_tokens": max_tokens}


def _config_value(run_dir: Path, key: str) -> str | None:
    """One `key = "value"` from the run's config.toml (no tomllib on 3.9)."""
    cfg = Path(run_dir) / "config.toml"
    if not cfg.exists():
        return None
    m = re.search(rf'^{key}\s*=\s*"(.+)"\s*$', cfg.read_text(), re.M)
    return m.group(1) if m else None


def _hf_home(run_dir: Path) -> str | None:
    return _config_value(run_dir, "hf_home")


def trained_from(repo: str | None, hf_home: str | None) -> dict | None:
    """§7.1 `trained_from` (D16): the weights the adapter was trained on, which
    are not the GGUF it was tested on. `precision` matters, not only as
    provenance: a picture Pigeon trained on 4-bit weights loses 10–30 of 200
    when served (D29). Read from the snapshot on disk, never typed."""
    if not repo:
        return None
    from .promptcheck import precision, revision
    out = {"repo": repo, "revision": revision(repo, hf_home), "precision": precision(repo, hf_home)}
    return {k: v for k, v in out.items() if v is not None}


def _mmproj_entry(run_dir: Path, check_hash: bool) -> tuple[dict | None, Path | None]:
    """The projector's name and hash, from the run's config.toml, and where
    it is — the packer carries it into the file."""
    cfg = run_dir / "config.toml"
    if not cfg.exists():
        return None, None
    m = re.search(r'^target_mmproj\s*=\s*"(.+)"\s*$', cfg.read_text(), re.M)
    if not m:
        return None, None
    p = Path(m.group(1))
    if not p.exists():
        return None, None
    return {"file": p.name, "sha256": sha256_file(p) if check_hash else None}, p
