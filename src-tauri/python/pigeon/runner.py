"""Asking a Pigeon and reading its answer — spec §8, the Runner's half.

The check ride is a Runner: it builds the prompt and reads the answer that
produce the score in report.json. If it differs from §8 anywhere, that score
describes something no host reproduces. So the ride (instructor/fly.py) uses
these functions, and the conformance kit's `prompt` and `answer` cases test
these same functions (pigeon/conformance.py). Standard library only.
"""
from __future__ import annotations

import json
import math
import re

_CITATION = re.compile(r"\[§([0-9]+)\]")


def fold(s: str) -> str:
    """§8.5: case-insensitive for ASCII letters only."""
    return "".join(c.lower() if c.isascii() else c for c in s)


def material_block(chunk: dict) -> str:
    """§9.3: one section as the model sees it."""
    return f"[§{chunk['ordinal']}] {chunk.get('title', '')}\n{chunk['text']}"


def split_grounded(row_text: str, chunks: list[dict]) -> tuple[dict, str] | None:
    """A Reader table row is one section's block, a blank line, then the
    question (`chunker.rs::material_block`). Return (that section, the question)
    so a ride can rebuild the prompt as a Runner does — or None when the row's
    passage is not one of the material's sections."""
    for c in chunks:
        block = material_block(c) + "\n\n"
        if row_text.startswith(block):
            return c, row_text[len(block):]
    return None


def fill(template: str, text: str, material: str | None = None) -> str:
    """§8.2 and §8.3: one left-to-right pass. The input is trimmed; text put into
    one slot is never scanned for slots again; `{image}` and one newline after it
    are dropped (the picture travels as its own part, first)."""
    out, i = [], 0
    while i < len(template):
        if template.startswith("{input}", i):
            out.append(text.strip())
            i += len("{input}")
        elif material is not None and template.startswith("{material}", i):
            out.append(material)
            i += len("{material}")
        elif template.startswith("{image}", i):
            i += len("{image}")
            if template.startswith("\n", i):
                i += 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


def messages(manifest: dict, text: str, *, system_prompt: str | None = None,
             sections: list[dict] | None = None, history: list[dict] | None = None,
             shots: list[dict] | None = None) -> list[dict]:
    """§8.1: the chat before the base's chat template is applied.

    `system_prompt` is system_prompt.txt, in any kind that carries one (D85);
    `shots` are shots.json's fixed turns, sent verbatim (D86, D87); `sections`
    are the chunks retrieval picked (grounded only); `history` is earlier
    `{user, pigeon}` exchanges, sent only under `history.shape = as_sent`.
    The newest user message is marked `image_first` for a picture Pigeon."""
    out: list[dict] = []
    if system_prompt is not None:
        out.append({"role": "system", "content": system_prompt})
    for turn in shots or []:
        out.append({"role": "user", "content": turn["user"]})
        out.append({"role": "assistant", "content": turn["pigeon"]})
    hist = manifest.get("history") or {}
    if hist.get("shape") == "as_sent" and history:
        for turn in history[-int(hist["max_turns"]):]:
            out.append({"role": "user", "content": turn["user"].strip()})
            out.append({"role": "assistant", "content": turn["pigeon"]})
    material = None
    if manifest.get("kind") == "grounded":
        material = "\n\n".join(material_block(c) for c in (sections or []))
    user = {"role": "user", "content": fill(manifest["prompt_template"], text, material)}
    if "image" in (manifest.get("inputs") or []):
        user["image_first"] = True
    out.append(user)
    return out


def cited(answer: str) -> list[int]:
    """§9.6: the section ordinals an answer cites, in first-seen order, once each."""
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def read_answer(contract: dict, raw: str, *, labels: list[str] | None = None, kind: str | None = None,
                skill_payload: dict | None = None, shown: list[int] | None = None) -> dict:
    """§8.5: the model's text as the answer the Pigeon promises.

    Returns {"kind": "label"|"json"|"recipe"|"text"|"off_contract", ...}. An
    `off_contract` answer is an answer, not an error, and nothing is repaired:
    no nearest label, no closing a `<do>` the model left open."""
    raw = raw.strip()
    otype = contract.get("type")
    if otype == "reasoned":
        at = raw.rfind("<do>")
        end = raw.find("</do>", at) if at >= 0 else -1
        if at < 0 or end < 0:
            return {"kind": "off_contract", "raw": raw}
        inner = read_answer({**contract, "type": contract.get("answer")}, raw[at + len("<do>"):end],
                            labels=labels, kind=kind, skill_payload=skill_payload, shown=shown)
        if inner["kind"] != "off_contract":
            inner["reasoning"] = raw[:at].strip()
        return inner
    if otype == "label":
        for label in labels or []:
            if fold(label) == fold(raw):
                return {"kind": "label", "value": label}
        return {"kind": "off_contract", "raw": raw}
    if otype == "json":
        try:
            obj = json.loads(raw)
        except ValueError:
            return {"kind": "off_contract", "raw": raw}
        if not isinstance(obj, dict):
            return {"kind": "off_contract", "raw": raw}
        if skill_payload is not None:
            art = render_concat(skill_payload, obj)
            if art is None:
                return {"kind": "off_contract", "raw": raw}
            return {"kind": "recipe", "value": obj, "artefact": art}
        return {"kind": "json", "value": obj}
    out = {"kind": "text", "value": raw}
    if kind == "grounded":
        out["unsupported_citations"] = [n for n in cited(raw) if n not in (shown or [])]
    return out


_KEY = re.compile(r"KEY\((.+)\)")


ACTIONS_VERSION = 1


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _round(v: float) -> int:
    """Spec §13.2: halves away from zero, as Swift's `.rounded()` does."""
    return int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)


def trim(crop) -> tuple[float, float, float, float]:
    """§13.2 step 0 (D70), the recorder's `Crop`: the start moved inside the
    window, then the width and height cut to fit. None is the whole window."""
    if crop is None:
        return 0.0, 0.0, 1.0, 1.0
    x, y = min(max(crop[0], 0), 1), min(max(crop[1], 0), 1)
    return x, y, min(crop[2], 1 - x), min(crop[3], 1 - y)


def frames_problem(frames) -> str | None:
    """Why a `frames` block cannot be followed (spec §13.2, D66, D69, D70,
    D72), or None. A host that cannot follow it must not play the Pigeon on a
    guess. Whether the kept part rounds to under a pixel needs the window:
    `frames_plan` checks that."""
    if not isinstance(frames, dict):
        return "it has no frames"
    stack, side, crop = frames.get("stack"), frames.get("max_side"), frames.get("crop")
    if not (_whole(stack) and 1 <= stack <= 4):
        return "frames.stack is not a whole number from 1 to 4"
    if not (_whole(side) and side >= 64):
        return "frames.max_side is not a whole number of at least 64"          # D77
    gap = frames.get("gap_ms")
    if gap is not None and not (_number(gap) and gap >= 0):
        return "frames.gap_ms is not a number of at least 0"                   # D76
    if stack > 1 and gap is None:
        return "frames.stack is above 1 with no gap_ms"                        # D72
    if crop is not None:
        if not (isinstance(crop, list) and len(crop) == 4 and all(_number(v) for v in crop)):
            return "frames.crop is not four numbers"
        _, _, w, h = trim(crop)
        # Plain double arithmetic, no tolerance, as Recorder.swift does it (D75):
        # [0.99, 0, 0.5, 1] trims to a hair above 1% and is followed.
        if w <= 0.01 or h <= 0.01:
            return "frames.crop keeps 1% or less of the window"
    return None


def frames_plan(window, frames) -> dict:
    """§13.2 steps 0 to 3, the recorder's `captureSize` and `cropped`, for a
    window of W0 x H0 pixels: {"result": "followed", "capture": [W, H],
    "kept": [left, top, width, height]}, or {"result": "skipped"}."""
    if frames_problem(frames):
        return {"result": "skipped"}
    W0, H0 = window
    x, y, w, h = trim(frames.get("crop"))
    k = min(1.0, frames["max_side"] / max(W0 * w, H0 * h))
    W, H = _round(W0 * k), _round(H0 * k)
    left, top = _round(x * W), _round(y * H)
    right, bottom = min(left + _round(w * W), W), min(top + _round(h * H), H)
    if right - left < 1 or bottom - top < 1:
        return {"result": "skipped"}
    return {"result": "followed", "capture": [W, H], "kept": [left, top, right - left, bottom - top]}


def actions_problem(raw: bytes | None, manifest: dict) -> str | None:
    """Why `actions.json` is skipped (§13.2), or None when it can be used."""
    if raw is None:
        return "there is no actions.json"
    try:
        acts = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "it is not JSON"
    if not isinstance(acts, dict):
        return "it is not a JSON object"
    if "image" not in (manifest.get("inputs") or []):
        return "this Pigeon takes no pictures"
    fv = acts.get("format_version")
    if not _whole(fv):
        return "it has no whole-number format_version"
    if fv > ACTIONS_VERSION:
        return f"it is actions.json format {fv}; this app reads up to {ACTIONS_VERSION}"
    if not isinstance(acts.get("keys", {}), dict):
        return "its 'keys' is not an object"
    return frames_problem(acts.get("frames"))


def usable_actions(raw: bytes | None, manifest: dict) -> dict | None:
    """`actions.json` as a Runner may use it, or None when it is skipped
    (§13.2): see `actions_problem`. A missing `keys` is an empty one."""
    if actions_problem(raw, manifest) is not None:
        return None
    return json.loads(raw.decode("utf-8"))


def listed_key(entry) -> bool:
    """D68: a key is listed only as an object with a whole-number `code` from 0
    to 127 (a macOS virtual key code). Another key's bad code does not matter."""
    return isinstance(entry, dict) and _whole(entry.get("code")) and 0 <= entry["code"] <= 127


def action_for(actions: dict | None, answer: dict) -> dict:
    """§13.2, in the engine's words (INTERFACE.md §6.2). `action` is the file's
    own declaration for a `KEY(name)` label, with the name as `key`;
    `not_pressed` is the file's row for a label it deliberately leaves out. Both
    are None otherwise: a host presses nothing the file does not list under `keys`."""
    out = {"action": None, "not_pressed": None}
    m = _KEY.fullmatch(answer.get("value") or "") if answer.get("kind") == "label" else None
    if actions is None or not m:
        return out
    declared = actions.get("keys", {}).get(m.group(1))
    if listed_key(declared):
        out["action"] = {"key": m.group(1), **declared}
    else:
        out["not_pressed"] = next((r for r in actions.get("not_pressed") or []
                                   if isinstance(r, dict) and r.get("answer") == answer["value"]), None)
    return out


def render_concat(payload: dict, recipe: dict) -> str | None:
    """§10.4 concat v1. None when the recipe names a part the table lacks."""
    out = [payload.get("head", "")]
    for key in payload["order"]:
        v = recipe.get(key)
        if v in (None, False, "", []):
            continue
        for name in (v if isinstance(v, list) else [v]):
            if not isinstance(name, str) or name not in payload["parts"].get(key, {}):
                return None
            out.append(payload["parts"][key][name])
    return payload.get("wrapper", "{parts}").replace("{parts}", "".join(out))
