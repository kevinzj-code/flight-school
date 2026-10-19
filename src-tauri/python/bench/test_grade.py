"""How a check-ride answer is graded. No model, no server.

    python3 -m pytest src-tauri/python/bench/test_grade.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from instructor import grade  # noqa: E402

NORM = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())  # noqa: E731  (fly._norm)
RECIPE = '{"eyes":"dots","mouth":"grin","brows":null,"extras":[]}'


def test_a_plan_with_no_grader_is_unchanged():
    assert grade.rule_for({}) is None
    assert grade.grade_row("penny", "penny", labels=["penny", "dime"], rule=None,
                           judge=True, norm=NORM) == (True, "exact")
    assert grade.grade_row("a long prose answer well over the exact threshold",
                           "something else", labels=None, rule=None,
                           judge=True, norm=NORM) == (False, "judge")


def test_rules_are_checked_before_labels():
    """label_set is a regex for 'one of: a, b, c' over output_shape, so a job
    that lists its own vocabulary there would be captured as a classifier by
    accident. The rule has to win."""
    rule = grade.Rule(name="t", grade=lambda e, a: (True, "ok"))
    passed, how = grade.grade_row("x", "y", labels=["eyes", "mouth"], rule=rule,
                                  judge=True, norm=NORM)
    assert (passed, how) == (True, "rules")


def test_a_rule_resolves_from_the_plan_and_grades():
    plan = {"grader": {"how": "rules", "rule": "facegrammar:grade_recipe",
                       "schema": "facegrammar:recipe_schema"}}
    rule = grade.rule_for(plan)
    assert rule is not None and rule.schema["properties"]["eyes"]["enum"]
    assert rule.grade(RECIPE, RECIPE)[0] is True
    wrong = RECIPE.replace('"grin"', '"smile"')
    passed, detail = rule.grade(RECIPE, wrong)
    assert passed is False and "mouth" in detail


def test_an_unresolvable_rule_raises_rather_than_falling_back_to_a_judge():
    """A grading mode that quietly degrades is how a deterministic score becomes
    a machine opinion with nobody noticing — the scorecard would still read
    graded_by_model: 0."""
    for bad in ({"how": "rules", "rule": "nosuchmodule:grade"},
                {"how": "rules", "rule": "facegrammar:no_such_function"},
                {"how": "rules"},
                {"how": "judge", "rule": "facegrammar:grade_recipe"}):
        with pytest.raises(ValueError if "rule" not in bad or bad.get("how") != "rules"
                           else Exception):
            grade.rule_for({"grader": bad})


def test_needs_judge_is_false_once_a_rule_exists():
    """quickbuild's second pass overwrites passed/how on every row, so this flag
    is what stops a rules verdict being replaced by a judged one."""
    assert grade.needs_judge({"output_shape": "one tidy sentence"}) is True
    assert grade.needs_judge({"output_shape": "exactly one of: penny, dime"}) is False
    assert grade.needs_judge({"output_shape": "one tidy sentence",
                              "grader": {"how": "rules",
                                         "rule": "facegrammar:grade_recipe"}}) is False


def test_the_adapter_and_the_control_cannot_be_graded_differently():
    """fly.py used to make this decision twice, with a comment on the second
    promising it matched the first."""
    src = (Path(__file__).resolve().parents[1] / "instructor/fly.py").read_text()
    assert src.count("grade_mod.grade_row(") == 2, "both passes call the one function"
    assert 'how = "exact"' not in src, "no copy of the decision survives in fly.py"
