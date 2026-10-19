"""THE CONTRACT between the model and the code.

The Instructor's only job is to fill this in. Code refuses anything that does
not match. Everything downstream — question generation, dataset building,
training, the card — reads a plan that has already passed through here.

Validation returns a *report*, not a boolean. Every field comes back marked:

    ok        the model got it right
    repaired  mechanically fixed (clamped a number, wrapped a bare string)
    rejected  needs judgment, so code will not guess

That distinction is the model-plans/code-executes line, drawn precisely.
A repair never invents content; if fixing a field would require deciding what
the person meant, it is a rejection and the plan does not run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The three fixed training presets. The Instructor picks time and depth; it
# never sets a learning rate or anything else under the hood. Iteration counts
# match Flight School's training.rs::preset_iters exactly.
PRESETS: dict[str, int] = {"quick": 200, "standard": 600, "thorough": 1200}

# The spec's example plan writes `"preset": "600"`. Both spellings are accepted
# and normalised to the name, because the number is the thing that varies
# between Flight School versions and the name is the thing that doesn't.
_PRESET_ALIASES = {"200": "quick", "600": "standard", "1200": "thorough",
                   "1,200": "thorough"}

CARTRIDGE_KINDS = ("behavior", "behavior+knowledge")
MODALITIES = ("text", "vision")

OK, REPAIRED, REJECTED = "ok", "repaired", "rejected"


@dataclass
class FieldResult:
    name: str
    status: str
    detail: str


@dataclass
class Report:
    plan: dict[str, Any]
    results: list[FieldResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(FieldResult(name, status, detail))

    @property
    def rejections(self) -> list[FieldResult]:
        return [r for r in self.results if r.status == REJECTED]

    @property
    def valid(self) -> bool:
        return not self.rejections

    @property
    def iters(self) -> int:
        return PRESETS[self.plan["preset"]]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "unnamed-job"


# ---------------------------------------------------------------------------
# field-by-field validation


def _need_str(rep: Report, plan: dict, key: str, *, max_len: int = 400) -> None:
    v = plan.get(key)
    if isinstance(v, str) and v.strip():
        cleaned = " ".join(v.split())
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len].rstrip()
            plan[key] = cleaned
            rep.add(key, REPAIRED, f"trimmed to {max_len} chars")
            return
        plan[key] = cleaned
        rep.add(key, OK, cleaned[:60])
        return
    rep.add(key, REJECTED, "missing or empty — this one has to come from the brief")


def _need_str_list(rep: Report, plan: dict, key: str, *, minimum: int) -> None:
    v = plan.get(key)
    if isinstance(v, str) and v.strip():
        # A bare string where a list belongs is a shape error, not a content
        # decision, so wrapping it is a legitimate mechanical repair.
        plan[key] = [" ".join(v.split())]
        rep.add(key, REPAIRED, "single string wrapped into a list")
        v = plan[key]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        rep.add(key, REJECTED, "must be a list of strings")
        return
    items = [" ".join(x.split()) for x in v if x and x.strip()]
    if len(items) < minimum:
        rep.add(key, REJECTED, f"needs at least {minimum}, got {len(items)}")
        return
    plan[key] = items
    rep.add(key, OK, f"{len(items)} item(s)")


def _validate_groups(rep: Report, plan: dict) -> None:
    groups = plan.get("question_groups")
    if not isinstance(groups, list) or not groups:
        rep.add("question_groups", REJECTED, "must be a non-empty list")
        return
    cleaned, problems = [], []
    for i, g in enumerate(groups):
        if not isinstance(g, dict):
            problems.append(f"group {i} is not an object")
            continue
        name = g.get("name")
        purpose = g.get("purpose")
        count = g.get("count")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"group {i} has no name")
            continue
        if not isinstance(purpose, str) or not purpose.strip():
            problems.append(f"group {name!r} has no purpose")
            continue
        try:
            count = int(count)
        except (TypeError, ValueError):
            problems.append(f"group {name!r} has a non-numeric count")
            continue
        if count < 1:
            problems.append(f"group {name!r} asks for {count} questions")
            continue
        cleaned.append({"name": name.strip(), "count": count, "purpose": purpose.strip()})
    if problems:
        rep.add("question_groups", REJECTED, "; ".join(problems[:3]))
        return
    plan["question_groups"] = cleaned
    total = sum(g["count"] for g in cleaned)
    rep.add("question_groups", OK, f"{len(cleaned)} groups, {total} questions")


def _validate_counts(rep: Report, plan: dict) -> None:
    """holdout_count and max_answer_tokens — both bounded by arithmetic."""
    groups = plan.get("question_groups")
    total = sum(g["count"] for g in groups) if isinstance(groups, list) and groups \
        and all(isinstance(g, dict) and isinstance(g.get("count"), int) for g in groups) else 0

    raw = plan.get("holdout_count")
    try:
        holdout = int(raw)
    except (TypeError, ValueError):
        rep.add("holdout_count", REJECTED, "must be a whole number")
        holdout = None

    if holdout is not None:
        if total == 0:
            rep.add("holdout_count", REJECTED, "cannot be checked — question_groups is invalid")
        elif holdout < 5:
            rep.add("holdout_count", REJECTED,
                    f"{holdout} is too few to be a check ride; ask for at least 5")
        elif holdout >= total:
            rep.add("holdout_count", REJECTED,
                    f"{holdout} held out of {total} total leaves nothing to train on")
        elif holdout > total * 0.4:
            capped = max(5, int(total * 0.4))
            plan["holdout_count"] = capped
            rep.add("holdout_count", REPAIRED,
                    f"{holdout} was over 40% of {total}; capped to {capped}")
        else:
            plan["holdout_count"] = holdout
            rep.add("holdout_count", OK, f"{holdout} of {total} held out")

    # max_answer_tokens is a CLAIM. It is sanity-bounded here and then
    # re-measured against the real tokenizer in preflight.py, which overrides
    # it. This field never decides a truncation on its own.
    raw = plan.get("max_answer_tokens")
    try:
        mat = int(raw)
    except (TypeError, ValueError):
        rep.add("max_answer_tokens", REJECTED, "must be a whole number")
        return
    if mat < 1:
        rep.add("max_answer_tokens", REJECTED, f"{mat} tokens is not an answer")
    elif mat > 2048:
        plan["max_answer_tokens"] = 2048
        rep.add("max_answer_tokens", REPAIRED, f"{mat} clamped to 2048")
    else:
        plan["max_answer_tokens"] = mat
        rep.add("max_answer_tokens", OK, f"{mat} tokens (a claim — re-measured at build)")


def _validate_preset(rep: Report, plan: dict) -> None:
    v = plan.get("preset")
    key = str(v).strip().lower() if v is not None else ""
    if key in _PRESET_ALIASES:
        plan["preset"] = _PRESET_ALIASES[key]
        rep.add("preset", REPAIRED, f"{v!r} read as {plan['preset']} "
                                    f"({PRESETS[plan['preset']]} iters)")
    elif key in PRESETS:
        plan["preset"] = key
        rep.add("preset", OK, f"{key} ({PRESETS[key]} iters)")
    else:
        rep.add("preset", REJECTED,
                f"{v!r} is not one of the three presets: quick / standard / thorough")


def _validate_kind(rep: Report, plan: dict) -> None:
    """cartridge_kind, knowledge_file, modality and image_corpus together.

    These four are one decision, so they are checked as one. An adapter nudges
    weights; it does not install memories. A plan that claims facts live in the
    weights is the single most expensive mistake this file can catch.
    """
    kind = str(plan.get("cartridge_kind", "")).strip().lower().replace("_", "+")
    kind = {"behaviour": "behavior", "behaviour+knowledge": "behavior+knowledge",
            "behavior + knowledge": "behavior+knowledge"}.get(kind, kind)
    if kind not in CARTRIDGE_KINDS:
        rep.add("cartridge_kind", REJECTED,
                f"{plan.get('cartridge_kind')!r} must be 'behavior' or 'behavior+knowledge'")
    else:
        plan["cartridge_kind"] = kind
        rep.add("cartridge_kind", OK, kind)

    kf = plan.get("knowledge_file")
    if kind == "behavior+knowledge":
        if not isinstance(kf, str) or not kf.strip():
            rep.add("knowledge_file", REJECTED,
                    "kind is behavior+knowledge but no knowledge file is named")
        else:
            plan["knowledge_file"] = kf.strip()
            rep.add("knowledge_file", OK, kf.strip())
    else:
        if kf:
            # Naming a knowledge file on a pure-behavior cartridge is a
            # contradiction about where facts live. Refuse; don't pick a side.
            rep.add("knowledge_file", REJECTED,
                    f"kind is {kind or 'behavior'} but a knowledge file "
                    f"({kf!r}) is named — decide which")
        else:
            plan["knowledge_file"] = None
            rep.add("knowledge_file", OK, "none (facts do not live in weights)")

    # modality is separate from kind on purpose: the spec's own connector
    # sorter is a *vision* job whose cartridge_kind is "behavior".
    modality = str(plan.get("modality", "text")).strip().lower()
    if modality not in MODALITIES:
        rep.add("modality", REJECTED, f"{modality!r} must be 'text' or 'vision'")
        return
    plan["modality"] = modality

    corpus = plan.get("image_corpus")
    if modality == "vision":
        if not isinstance(corpus, str) or not corpus.strip():
            rep.add("image_corpus", REJECTED,
                    "a vision job needs an image corpus — the model plans for "
                    "images, it cannot invent them")
        else:
            plan["image_corpus"] = corpus.strip()
            rep.add("image_corpus", OK, corpus.strip())
        rep.add("modality", OK, "vision (needs mmproj at check ride)")
    else:
        plan["image_corpus"] = None
        rep.add("modality", OK, "text")


def _tidy_text(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, (str, int, float)) else ""


def _validate_presentation(rep: Report, plan: dict[str, Any]) -> None:
    """The plan card's words are read, never trained on, so they are tidied
    and never rejected. A malformed part is dropped and named, and the card
    falls back to the pipeline fields (one_line, example_questions) for it."""
    for key in ("graduation_goal", "headline", "big_idea"):
        if key not in plan:
            continue
        text = _tidy_text(plan[key])
        if key == "headline":
            text = text.rstrip(".")
        if not text:
            plan.pop(key)
            rep.add(key, REPAIRED, "blank or not text, dropped")
        elif text != plan[key]:
            plan[key] = text
            rep.add(key, REPAIRED, "tidied")
        else:
            rep.add(key, OK, f"{len(text.split())} words")

    if "off_syllabus" in plan:
        items = plan["off_syllabus"] if isinstance(plan["off_syllabus"], list) else []
        kept = []
        for it in items:
            if isinstance(it, dict) and _tidy_text(it.get("topic")):
                kept.append({"topic": _tidy_text(it.get("topic")), "instead": _tidy_text(it.get("instead"))})
        if kept:
            if kept != plan["off_syllabus"]:
                rep.add("off_syllabus", REPAIRED, f"{len(kept)} kept of {len(items)}")
            else:
                rep.add("off_syllabus", OK, f"{len(kept)} limits")
            plan["off_syllabus"] = kept[:3]
        else:
            plan.pop("off_syllabus")
            rep.add("off_syllabus", REPAIRED, "no readable limits, dropped")

    if "pop_quiz" in plan:
        pq = plan["pop_quiz"] if isinstance(plan["pop_quiz"], dict) else {}
        asks, replies = _tidy_text(pq.get("asks")), str(pq.get("replies", "")).strip()
        if asks and replies:
            if {"asks": asks, "replies": replies} != plan["pop_quiz"]:
                rep.add("pop_quiz", REPAIRED, "tidied")
            else:
                rep.add("pop_quiz", OK, "")
            plan["pop_quiz"] = {"asks": asks, "replies": replies}
        else:
            plan.pop("pop_quiz")
            rep.add("pop_quiz", REPAIRED, "needs both a question and a reply, dropped")


def validate(raw_plan: dict[str, Any]) -> Report:
    """Validate and mechanically repair a plan. Never guesses at meaning."""
    plan = json.loads(json.dumps(raw_plan))  # deep copy; the caller's dict is untouched
    rep = Report(plan=plan)

    name = plan.get("job_name")
    if isinstance(name, str) and name.strip():
        slug = slugify(name)
        if slug != name:
            plan["job_name"] = slug
            rep.add("job_name", REPAIRED, f"{name!r} → {slug!r}")
        else:
            rep.add("job_name", OK, slug)
    else:
        rep.add("job_name", REJECTED, "missing")

    _need_str(rep, plan, "one_line")
    _validate_kind(rep, plan)
    _need_str(rep, plan, "input_shape")
    _need_str(rep, plan, "output_shape")
    _need_str_list(rep, plan, "style_rules", minimum=1)
    _validate_groups(rep, plan)
    _need_str_list(rep, plan, "example_questions", minimum=1)
    _need_str_list(rep, plan, "grading_rubric", minimum=1)
    _validate_counts(rep, plan)
    _validate_preset(rep, plan)

    _validate_presentation(rep, plan)

    unknown = set(plan) - set(FIELDS) - set(PRESENTATION)
    if unknown:
        # Extra keys are carried, not deleted — Flight School's own datasets
        # preserve unrecognised fields, and so does this. But they are named,
        # so a hallucinated field never passes unnoticed.
        rep.add("extra fields", REPAIRED, "carried through: " + ", ".join(sorted(unknown)))
    return rep


FIELDS = (
    "job_name", "one_line", "cartridge_kind", "knowledge_file", "modality",
    "image_corpus", "input_shape", "output_shape", "style_rules",
    "question_groups", "example_questions", "grading_rubric", "holdout_count",
    "max_answer_tokens", "preset",
)

#: What the head of school reads on the plan card. The planner writes them and
#: the validator tidies them, but a plan is complete without them: an imported
#: plan, or one from before the card existed, still runs.
PRESENTATION = ("graduation_goal", "headline", "big_idea", "off_syllabus", "pop_quiz")


def load_plan(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("a plan must be a JSON object")
    return data


def save_plan(plan: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: plan[k] for k in FIELDS if k in plan}
    ordered.update({k: v for k, v in plan.items() if k not in FIELDS})
    path.write_text(json.dumps(ordered, indent=2) + "\n")
    return path


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the plan object out of whatever the model actually said.

    Small models fence their JSON, prepend "Here's the plan:", or trail a
    sentence. Recovering the object from that is parsing, not judgment, so it
    belongs in code. If nothing parses, this returns None and the caller says
    so — it never repairs JSON by rewriting it.
    """
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [f.strip() for f in fenced]
    candidates.append(text.strip())
    start = text.find("{")
    if start != -1:
        # Scan for the matching brace, ignoring braces inside strings.
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
