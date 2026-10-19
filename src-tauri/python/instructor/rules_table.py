"""Grading rules for jobs made in the two-column table (Milestone 5).

A reasoning job answers `<think>…</think>` then `<do>ACTION</do>`. Only the
action is graded — the last `<do>` in the reply, matched exactly, the same way a
plain function answer is — and the reasoning is never scored (ONE-IN-ONE-OUT.md
§4). A reply with no `<do>` has no answer."""
import re

_DO = re.compile(r"<do>\s*(.*?)\s*</do>", re.S | re.I)


def last_do(text: str) -> str | None:
    found = _DO.findall(text or "")
    return found[-1].strip() if found else None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def reasoned(expected: str, actual: str) -> tuple[bool, str]:
    """Spec §8.5 `reasoned`, as every Runner reads it: the content of the LAST
    `<do>`, closed by a `</do>` after it, trimmed, then the label rule (ASCII
    case-insensitive, no repair). An unclosed last `<do>` is off_contract even
    when an earlier one closed; the old regex took the earlier one."""
    from pigeon.runner import read_answer
    want = last_do(expected) or expected.strip()
    got = read_answer({"type": "reasoned", "answer": "label"}, actual, labels=[want])
    if got["kind"] == "off_contract":
        return False, "off contract: no complete <do>, or not the label"
    return True, f"<do> {got['value']}"


def plan_steps(expected: str, actual: str) -> tuple[bool, str]:
    """A plan (Patch's planner, `{"about", "steps"}`): the answer is a JSON
    object, read as every Runner reads JSON (§8.5, no repair), and its steps —
    skills, ids and inputs, in order — equal the expected plan's. `about` is
    the runway, words of the model's own, and is not graded."""
    import json
    from pigeon.runner import read_answer
    got = read_answer({"type": "json"}, actual)
    if got["kind"] != "json":
        return False, "off contract: not a JSON object"
    want = json.loads(expected)
    if got["value"].get("steps") != want.get("steps"):
        skills = [s.get("skill") for s in got["value"].get("steps") or [] if isinstance(s, dict)]
        same = skills == [s["skill"] for s in want.get("steps") or []]
        return False, "the right skills, other inputs" if same else f"skills {skills}"
    return True, f"{len(want.get('steps') or [])} steps"
