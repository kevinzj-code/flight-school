"""How a check-ride answer is graded — the decision, in one place.

`fly.py` made this decision twice, once for the adapter pass and once for the
control, with a comment on the second promising it matched the first. Two copies
of a rule that MUST agree is how they stop agreeing, so both now call
`grade_row`.

It also adds a third mode. Today an answer is exact-matched when the job has a
fixed label set or the expected answer is 24 characters or less, and is otherwise
sent to a model judge. Structured output fits neither: an SVG is ~380 characters
and a parts recipe ~70, so both fall to a judge that cannot check structure at
all. A job whose answer is machine-checkable declares a RULE and is graded by it.

Why rules are checked BEFORE labels: `fly.label_set` is a regex for
"one of: a, b, c" over the plan's `output_shape`, so a job that happens to list
its vocabulary in that sentence would be captured as a classifier by accident.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

#: Where a bare "module:function" reference is resolved from. This directory is
#: already on sys.path for quickbuild and for the bench tests.
_HERE = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Rule:
    name: str
    grade: Callable[[str, str], tuple[bool, str]]
    schema: dict[str, Any] | None = None


#: The only modules a plan may name a rule in. A plan can arrive in a folder
#: someone downloaded (`lesson.json`, importset.py), so "any module, any
#: attribute, or any .py path on disk" was a way for a dataset to run code on
#: the check ride: `shutil:rmtree` with a row's own answer as the argument. A
#: rule is now one of the app's own graders, by name, and nothing else.
RULE_MODULES = ("instructor.rules_table", "facegrammar", "pigeonlook", "iconrules", "patchmotion")
_RULE_ATTR = re.compile(r"^(grade\w*|reasoned|\w*_schema)$")


def _resolve(ref: str) -> Any:
    """`module:attr`, where module is one of RULE_MODULES and attr is a grader
    or a schema. Never a path, never an arbitrary module."""
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(f"grader reference {ref!r} must be 'module:function'")
    where, attr = ref.rsplit(":", 1)
    if where not in RULE_MODULES:
        raise ValueError(f"grader module {where!r} is not one of the app's own graders: {', '.join(RULE_MODULES)}")
    if not _RULE_ATTR.match(attr):
        raise ValueError(f"grader {attr!r} is not a grading function or a schema")
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    mod = importlib.import_module(where)
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise ValueError(f"{where} has no {attr!r}")
    return fn


def rule_for(plan: dict) -> Rule | None:
    """The plan's declared rule, resolved, or None when it declares none.

    RAISES on an unresolvable reference. A grading mode that quietly degrades to
    a judge is how a deterministic score becomes a machine opinion with nobody
    noticing — and the scorecard would still say `graded_by_model: 0`.
    """
    spec = plan.get("grader")
    if not spec:
        return None
    if not isinstance(spec, dict) or spec.get("how") != "rules":
        raise ValueError(f"plan.grader must be an object with how='rules', got {spec!r}")
    ref = spec.get("rule")
    if not ref:
        raise ValueError("plan.grader needs a 'rule' reference")
    grade = _resolve(ref)
    schema = None
    if spec.get("schema"):
        schema = _resolve(spec["schema"])()
    return Rule(name=ref, grade=grade, schema=schema)


def needs_judge(plan: dict) -> bool:
    """Whether this job's prose answers need a model's opinion.

    `quickbuild.stage_fly` decides this and then OVERWRITES `passed` and `how`
    on every row from its own judging pass — so a rules-graded run that did not
    consult this would have its deterministic verdicts replaced by judged ones.
    """
    from . import fly  # local: fly imports this module's caller graph at runtime
    return fly.label_set(plan) is None and rule_for(plan) is None


def grade_row(expected: str, actual: str, *, labels, rule: Rule | None,
              judge: bool, norm: Callable[[str], str]) -> tuple[bool, str]:
    """`(passed, how)` where how is "rules" | "exact" | "judge".

    `judge` rows come back False here and are decided in the second pass."""
    if rule is not None:
        passed, _detail = rule.grade(expected, actual)
        return passed, "rules"
    if labels:
        # Spec §8.5, as every Runner reads it: the trimmed answer matches a line
        # of the label set case-insensitively (ASCII) and nothing is repaired.
        # The ride used to strip all punctuation first, so "KEY(Up)." passed
        # here and came back off_contract everywhere else.
        from pigeon.runner import fold, read_answer
        got = read_answer({"type": "label"}, actual, labels=list(labels))
        return got["kind"] == "label" and fold(got["value"]) == fold(expected.strip()), "exact"
    if len(norm(expected)) <= 24:
        # A label answer buried in a sentence is still the wrong SHAPE for a
        # cartridge whose whole job was the shape, so this stays strict.
        return norm(actual) == norm(expected), "exact"
    if judge:
        return False, "judge"
    return norm(actual) == norm(expected), "exact"
