"""Put a finished run into one file.

Spec §3.1: a ZIP with a different extension, store mode (the adapter is
already dense), fixed layout, files at the root, no wrapper folder. Spec §3.4:
the training data stays on the machine that trained it — `report.json`
describes the set, it never carries it.
"""

from __future__ import annotations

import json
import random
import shutil
import struct
import zipfile
import zlib
from pathlib import Path

from . import registry as registry_mod
from .manifest import build_manifest


# ---------------------------------------------------------------------------
# examples/ — three to five held-out inputs with their right answers


def pick_examples(holdout: list[dict], n: int = 4, seed: int = 11) -> list[dict]:
    """Stratified across groups, so the samples show the range of the job
    rather than four of the same thing. Deterministic for a seed so "swap"
    means a different seed, not a different rule."""
    n = max(3, min(5, n))
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = {}
    for row in holdout:
        by_group.setdefault(row.get("group") or "", []).append(row)
    for rows in by_group.values():
        rng.shuffle(rows)
    picked: list[dict] = []
    names = sorted(by_group)
    while len(picked) < n and any(by_group.values()):
        for g in names:
            if by_group[g] and len(picked) < n:
                picked.append(by_group[g].pop())
    return picked


# ---------------------------------------------------------------------------
# README.md — prefilled, always editable, never the data


def draft_readme(manifest: dict, facts: dict) -> str:
    plan, flight = facts["plan"], facts.get("flight")
    kind_words = {"classifier": "a sorter", "freeform": "a format" if not facts["state"].get("creative") else "a voice",
                  "persona": "a voice", "extractor": "an extractor", "router": "a router",
                  "grounded": "a reader"}[manifest["kind"]]
    lines = [f"# {manifest['name']}", "", "## What it does", "",
             plan.get("one_line") or "", "",
             f"Takes {plan.get('input_shape') or 'one input'}; gives {plan.get('output_shape') or 'one answer'}.", ""]
    if facts.get("labels"):
        lines += ["The only answers it gives: " + ", ".join(f"`{l}`" for l in facts["labels"]) + ".", ""]
    if manifest.get("skill"):
        sk = manifest["skill"]
        lines += ["## How to use it", "",
                  f"This Pigeon answers with a **recipe**, not the finished thing — "
                  f"a small JSON object naming "
                  + ", ".join(f"`{f}`" for f in manifest["output_contract"].get("fields") or [])
                  + ". The file carries **`skill/skill.json`**, which holds every piece "
                    "a recipe can name.",
                  "",
                  "Rendering is concatenation, not code: take `head`, then the fragment for "
                  "each named part in the order `order` gives, and put them inside `wrapper`. "
                  "Ten lines in any language, and nothing from this file is ever executed.",
                  "",
                  f"An app that does not recognise **{sk['name']} v{sk['version']}** "
                  f"must refuse this file rather than guess — rendering a recipe with a "
                  f"different parts table produces a well-formed drawing of the wrong thing.",
                  ""]
    lines += ["## What it's bad at", ""]
    misses = [r for r in (flight or {}).get("rows", []) if not r.get("passed")]
    if facts.get("source"):
        lines += ["This lesson was **checked by asking, not scored**: its practice answers were graded by hand, "
                  "and the check ride put the same questions to the stock LLM and the trained one side by side. "
                  "There is no pass/fail number inside this file, and the examples below are samples from what it "
                  "practiced on, not a test.", ""]
    elif flight and not misses:
        lines += [f"On its test it missed nothing ({flight['passed']}/{flight['total']}). That is one test on one machine; a run of new inputs will find edges this one did not.", ""]
    elif misses:
        lines += [f"On its test it missed {len(misses)} of {flight['total']}. The kind of case it got wrong:", ""]
        for r in misses[:3]:
            width = 140 if manifest["output_contract"]["type"] == "json" else 40
            lines += [f"- _{str(r.get('question', ''))[:80]}_\n"
                      f"  - wanted `{str(r.get('expected', ''))[:width]}`\n"
                      f"  - gave `{str(r.get('actual', ''))[:width]}`"]
        lines += [""]
    else:
        lines += ["Not tested yet.", ""]
    if manifest.get("material"):
        mat = manifest["material"]
        how = ("all of it is put in front of the LLM every time you ask"
               if mat.get("retrieval") == "whole"
               else "the app searches it and puts the few relevant sections in front of the LLM")
        lines += ["## What it answers from", "",
                  f"This file carries the material itself — {mat.get('sections')} sections"
                  + (f" from {', '.join(mat.get('files') or [])}" if mat.get("files") else "")
                  + f" — and {how}.", "",
                  "That is where the answers come from. The training taught it how to answer "
                  "from material it is given, not what the material says, so handing it a "
                  "different document of the same kind is the normal way to use it. Ask it "
                  "something the material does not cover and the honest answer is that it "
                  "does not cover it.", ""]
    elif manifest.get("skill"):
        lines += ["It picks parts; it does not invent them. Everything it can draw is in "
                  "`skill/skill.json`, and a name it has never met gets the nearest parts "
                  "it knows rather than a new idea.", ""]
    else:
        lines += ["It is prompt compression, not knowledge: it makes this one job come out in this one shape, reliably. It does not know anything the LLM it rides did not already know.", ""]
    lines += ["## What it was trained on", ""]
    if facts.get("source"):
        lines += [f"{manifest['example_count']} practice pairs, every one graded and approved by hand. The practice set stays on the machine that trained it; this file carries {len(facts['source'].get('examples', []))} approved samples, not the data.", ""]
    else:
        made = facts["state"].get("practice_source") or (
            "from a folder of labeled photos" if facts["vision"]
            else "by a local LLM from a plain-words brief")
        lines += [f"{manifest['example_count']} practice examples, {made}, audited before "
                  "training. The practice set stays on the machine that trained it; this "
                  "file carries the scorecard, not the data.", ""]
    lines += [
              f"Needs `{manifest['base_model']['file']}`" + (" and its vision projector" if facts["vision"] else "") + f" — {kind_words}, trained with {manifest['trained_with']} on {manifest['trained_on']}.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# icon.png — a 512×512 default, standard library only


def default_icon_png(initials: str, size: int = 512) -> bytes:
    """A flat ink square with the initials blocked out of a 5×7 pixel font.
    Not pretty; present. A person can replace it with any PNG in the last
    step, and the app scales it."""
    ink, paper = (0x16, 0x28, 0x3A), (0xE5, 0xF1, 0xF7)
    font = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
        "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
        "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    }
    text = "".join(ch for ch in initials.upper() if ch in font)[:2] or "P"
    cell = size // 16                       # glyph pixel size
    gw, gh = 5 * cell, 7 * cell
    total_w = len(text) * gw + (len(text) - 1) * cell
    x0, y0 = (size - total_w) // 2, (size - gh) // 2
    rows = []
    for y in range(size):
        row = bytearray([0])                # filter byte: none
        for x in range(size):
            px = ink
            gx = x - x0
            if y0 <= y < y0 + gh and 0 <= gx < total_w:
                i, off = divmod(gx, gw + cell)
                if i < len(text) and off < gw:
                    if font[text[i]][(y - y0) // cell][off // cell] == "1":
                        px = paper
            row += bytes(px)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# the container


def build_report(facts: dict) -> dict:
    """The scorecard, carried unchanged — control included — plus the build
    facts. This is where "36/36" lives, so the number travels with the file
    instead of living in a folder someone will delete. No rows of training
    data: the check ride's own rows are the held-out test, not the practice
    set, and even those are trimmed to what a reader needs."""
    st, flight = facts["state"], facts.get("flight")
    report = {
        "check_ride": None if not flight else {
            "passed": flight.get("passed"), "total": flight.get("total"),
            "control_passed": flight.get("control_passed"),
            "graded_by_model": flight.get("graded_by_model"),
            "paired": flight.get("paired"),
            "significance": flight.get("significance"),
            "power": flight.get("power"),
            # Where the numbers were measured. `unrecorded` is not a formality:
            # a run folder assembled by hand can carry any `flight` block, and
            # one measured in a runtime nobody serves with reads exactly like a
            # real check ride until someone runs the file. Flight School's own
            # rides stamp "llama.cpp" (instructor/fly.py::Flight.runtime).
            "runtime": flight.get("runtime") or "unrecorded",
            # D98: the engine that answered every row, when one did.
            **({"engine": flight["engine"]} if flight.get("engine") else {}),
            # §12.1: the file the ride ran on, and how it decoded. They must
            # equal the manifest's (base_model.sha256, temperature, constrained).
            "file_sha256": facts.get("base_sha256"),
            "temperature": flight.get("temperature", 0),
            "passes": 1,
            "constrained": bool(flight.get("constrained")),
            "history": "none",
            "by_group": flight.get("by_group"),
            # A Reader's deterministic check: answers that lean on the expected
            # answer's sentence (instructor/grounding.py `source_check`). An
            # extra field; readers ignore what they do not know.
            **({"same_source": flight["same_source"]} if flight.get("same_source") else {}),
            "rows": [{"question": r.get("question"), "expected": r.get("expected"),
                      "actual": r.get("actual"), "passed": r.get("passed"), "how": r.get("how"),
                      **({"same_source": r["same_source"]} if r.get("same_source") else {})}
                     for r in flight.get("rows", [])],
        },
        # §12.1 (D21, D30): one entry per ride of the same test. Flight School
        # rides once, in llama.cpp; a ride in the training runtime is next.
        "check_rides": check_rides(facts),
        "practice_set": {
            "rows": st.get("rows"), "split": st.get("dataset"),
            # Where the rows actually came from. The two defaults are Flight
            # School's own lanes; a run that was built another way says so in
            # `state.practice_source` and that wins, because a report.json is a
            # claim about provenance and a wrong one is worse than none. Found
            # by packing a hand-drawn, hand-reviewed corpus and reading back
            # "drafted by a local LLM from a brief".
            "source": st.get("practice_source")
                      or ("a folder of labeled photos" if facts["vision"]
                          else "drafted by a local LLM from a brief"),
            "audit": st.get("audit"),
            "drafting": st.get("drafting"),
        },
        "preflight": st.get("preflight"),
        "training": st.get("training"),
        "conversion": st.get("conversion"),
        "timings": st.get("timings"),
        "notes": st.get("notes"),
    }
    return report


def serving_trails(llama: dict, mlx: dict) -> str | None:
    """A sentence for the Package step when serving clearly trails training,
    or None. "Clearly" is the bake-off's measured floor: more than 5 of the
    exam and more than 5% of it (one base moved by up to 5 of 30 between seeds)."""
    gap = (mlx.get("passed") or 0) - (llama.get("passed") or 0)
    total = llama.get("total") or 0
    if total and gap > 5 and gap > 0.05 * total:
        return (f"served in llama.cpp it scored {llama['passed']} of {total}, but {mlx['passed']} where it "
                f"trained (MLX): {gap} answers are lost between training and serving")
    return None


def check_rides(facts: dict) -> list[dict]:
    """§12.1 (D21, D30): one entry per ride of the same test. The run records
    them (quickbuild.py: llama.cpp, then MLX where the adapter trained); a run
    from before that records only its llama.cpp ride."""
    flight = facts.get("flight")
    if not flight:
        return []
    rides = [dict(r) for r in flight.get("check_rides") or [{
        "runtime": flight.get("runtime") or "unrecorded", "passed": flight.get("passed"),
        "total": flight.get("total"), "control_passed": flight.get("control_passed"),
        "temperature": flight.get("temperature", 0), "constrained": bool(flight.get("constrained"))}]]
    for r in rides:
        if r.get("runtime") != "mlx":
            r["file_sha256"] = facts.get("base_sha256")
    return rides


def example_stem(i: int, ex: dict, vision: bool) -> str:
    """The name an example's picture is packed under (examples/NNN.<ext>)."""
    if vision and ex.get("image"):
        return f"examples/{i:03d}{Path(ex['image']).suffix.lower()}"
    return f"examples/{i:03d}.txt"


def build_prompt_check(run_dir: Path, manifest: dict, facts: dict, example: dict) -> tuple[dict | None, str | None]:
    """`prompt_check` for the first example, rendered by the TRAINING code path
    (pigeon/promptcheck.py) in the venv that trained the run: mlx-vlm for
    pictures, mlx-lm for text. Returns (prompt_check, None) or (None, why)."""
    import subprocess
    from .manifest import _config_value
    repo = (facts.get("trained_from") or {}).get("repo") or (facts["state"].get("base") or {}).get("student_repo")
    base_json = Path(run_dir) / "base.json"
    if not repo and base_json.exists():
        repo = json.loads(base_json.read_text()).get("student_repo")
    venv = _config_value(run_dir, "vision" if facts["vision"] else "train")
    if not repo or not venv or not Path(venv).exists():
        return None, "the run has no training venv or base repo to render the training prompt with"
    question = str(example.get("question", ""))
    if facts["vision"]:
        if not example.get("image") or not Path(example["image"]).exists():
            return None, "the first example has no picture on disk"
        req = {"mode": "picture", "image": str(example["image"]), "text": question}
        if facts.get("projector"):
            from .picturesize import limits
            req["limits"] = limits(facts["projector"])           # D80: the projector's, as training used
        pc_input = {"text": question.strip(), "image": example_stem(1, example, True)}
    else:
        req = {"mode": "text", "user": question}
        pc_input = question.strip()
        if facts.get("material"):
            pc_input = _grounded_input(facts, question)
            if pc_input is None:
                return None, "the first example's passage is not one of the material's sections"
    req.update(repo=repo, hf_home=_config_value(run_dir, "hf_home"))
    script = Path(__file__).with_name("promptcheck.py")
    try:
        out = subprocess.run([venv, str(script), json.dumps(req)], capture_output=True, text=True, timeout=600)
        got = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as e:
        return None, f"the training prompt could not be rendered ({e})"
    if "canonical" not in got:
        return None, got.get("error") or "the training prompt could not be rendered"
    return {"input": pc_input, "canonical": got["canonical"]}, None


def count_served(run_dir: Path, facts: dict, example: dict) -> int:
    """The check picture's tokens in llama.cpp, with the run's base and
    projector (D80). Raises when it cannot be counted: a picture Pigeon whose
    size cannot be checked is not packed."""
    try:
        from instructor.config import load
        from instructor.engine import LlamaServer, ServerSpec
        from .picturesize import count_in_llama
        cfg = load(Path(run_dir) / "config.toml")
        projector = facts.get("projector") or cfg.target_mmproj
        with LlamaServer(cfg, ServerSpec(model=cfg.target_gguf, mmproj=Path(projector), ctx=4096),
                         log=Path(run_dir) / "export" / "picture-count.log") as srv:
            return count_in_llama(srv, example["image"], str(example.get("question", "")))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"its check picture could not be counted in llama.cpp, so its size cannot be checked (D80): {e}")


def _grounded_input(facts: dict, question: str) -> dict | None:
    """§8.6 for a grounded Pigeon: `{input, sections}`. A Reader row's question
    is the section block above the question (`chunker.rs::material_block`);
    split it back into the two the Runner fills."""
    from .runner import split_grounded
    chunks = json.loads((Path(facts["material_dir"]) / "chunks.json").read_text()).get("chunks", [])
    found = split_grounded(question, chunks)
    return {"input": found[1].strip(), "sections": [found[0]["ordinal"]]} if found else None


def example_pool(run_dir: Path, facts: dict) -> list[dict]:
    """Where examples come from: the held-out set for a run (a test the
    training never saw), the approved pairs for a lesson (samples of what it
    practiced on — the README says which)."""
    src = facts.get("source")
    if src:
        return [{"question": e.get("question", ""), "answer": e.get("answer", ""),
                 "group": e.get("group") or e.get("category") or ""} for e in src.get("examples", [])]
    holdout_path = Path(run_dir) / "export" / "holdout.jsonl"
    if not holdout_path.exists():
        return []
    return [json.loads(l) for l in holdout_path.read_text().splitlines() if l.strip()]


def recorded_run(run_dir: Path) -> bool:
    """Whether a run's pictures came from Flight School's recorder: a run with
    a recording, or a table that has recorded sessions."""
    if (run_dir / "recording.json").is_file():
        return True
    try:
        return bool(json.loads((run_dir / "table.json").read_text()).get("sessionsIn"))
    except (OSError, ValueError, AttributeError):
        return False


def pack_run(run_dir: Path | str, out_path: Path | str, *, name: str | None = None,
             readme: str | None = None, icon_png: bytes | None = None,
             look: dict | None = None,
             example_seed: int = 11, example_count: int = 4,
             author: str | None = None, trained_with: str = "Flight School",
             does: str | None = None, input_hint: str | None = None, output_noun: str | None = None,
             talks_like: str | None = None, revision: int = 1,
             llama_server: Path | str | None = None, model_dirs=None, warnings: list | None = None,
             prompt_check_builder=None, registry: dict | None = None, picture_counter=None,
             ride_copy: bool = False) -> dict:
    """Write `<out_path>` and return the manifest. Raises on anything that
    would make the file a lie — a missing adapter, an unreadable base, a card
    whose text would have to be cut to fit.

    A grounded run's material is embedded first (embed.py) when the server
    and the model are at hand; when they are not, the file is packed with the
    lexical search and `warnings` says why.

    `registry` is the app's sealed base registry, already opened under its
    registry key list (host.open_bundled); a projector it lists for the base
    is left out of the file (§13.1). None carries every projector.

    `ride_copy` writes the copy the engine rides before the run has a check
    ride (DEV-6): no report, whatever an earlier attempt left in state.json."""
    run_dir, out_path = Path(run_dir), Path(out_path)
    from .manifest import material_dir as _material_dir
    mat = _material_dir(run_dir)
    if mat is not None:
        from . import embed
        _, why = embed.ensure_embeddings(mat, llama_server=llama_server, model_dirs=model_dirs or [],
                                         log=run_dir / "embed-server.log")
        if why and warnings is not None:
            warnings.append(f"packed with keyword-overlap-v2, not hybrid-v1: {why}")
    manifest, facts = build_manifest(run_dir, name=name, author=author, trained_with=trained_with,
                                     does=does, input_hint=input_hint, output_noun=output_noun,
                                     talks_like=talks_like, revision=revision)
    if facts.get("card_problems"):
        raise ValueError("the card cannot be packed: " + "; ".join(facts["card_problems"]))
    if ride_copy:
        facts["flight"] = None
    state = facts["state"]

    adapter = Path(state.get("adapter") or "")
    if not adapter.exists():
        raise FileNotFoundError("the run has no adapter file to package — it has not finished training")

    examples = pick_examples(example_pool(run_dir, facts), n=example_count, seed=example_seed)
    facts["base_sha256"] = manifest["base_model"].get("sha256")
    facts["trained_from"] = manifest.get("trained_from")
    # §12.1: say so when serving clearly trails where it trained.
    rides = check_rides(facts)
    mlx = next((r for r in rides if r.get("runtime") == "mlx"), None)
    if mlx and warnings is not None:
        lost = serving_trails(rides[0], mlx)
        if lost:
            warnings.append(lost)
    # §8.6 (D19, D28): one input and the exact prompt training built from it.
    # A Writer MUST write it; when it cannot be built (no venv, no model on
    # disk) the file still packs, and `warnings` says why it carries none.
    if examples:
        pc, why = (prompt_check_builder or build_prompt_check)(run_dir, manifest, facts, examples[0])
        if pc:
            manifest["prompt_check"] = pc
            # D80: the check picture is counted in llama.cpp, where it is
            # served, and the pack refused when training made it another size.
            import re as _re
            trained = _re.search(r"<image:(\d+)>", pc["canonical"])
            if trained and facts["vision"]:
                served = (picture_counter or count_served)(run_dir, facts, examples[0])
                from .picturesize import mismatch
                problem = mismatch(int(trained.group(1)), served)
                if problem:
                    raise ValueError(problem)
        elif warnings is not None:
            warnings.append(f"no prompt_check: {why}")

    labels = facts["labels"]
    readme = readme if readme is not None else draft_readme(manifest, facts)
    if icon_png is None:
        # No PNG chosen: draw the look the package step showed, or the one the
        # name picks — never the initials square, which every Pigeon shared.
        from pigeonlook import random_look, render_png
        icon_png = render_png(look or random_look(manifest["name"]))
    icon = icon_png

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("pigeon.json", json.dumps(manifest, indent=2) + "\n")
        z.write(adapter, "adapter.lora.gguf")
        # A picture Pigeon names its projector by hash in base_model.mmproj
        # either way. It CARRIES it only when the projector is not stock
        # (§13.1): one the app's sealed registry lists for the base is
        # installed with the base or downloaded from the registry, so the
        # file stays small. With no registry, every projector is carried.
        if facts.get("projector") and not registry_mod.stock_projector(registry, manifest["base_model"]):
            z.write(facts["projector"], "projector.gguf")
        # What a picture Pigeon made by recording PRESSES: each key answer's
        # key code and how long it was held, and how its pictures were made
        # (Flight School writes it from the table; Pigeon's docs/PLAY-LOOP.md).
        # Only a vision Pigeon plays, so only a vision Pigeon carries it.
        if facts.get("projector") and (run_dir / "actions.json").is_file():
            # Only for pictures the recorder made (spec §13.2, D71): imported
            # pictures were not captured from a window, so nothing can say how
            # to make a matching one live. Such a Pigeon answers, but is not played.
            if not recorded_run(run_dir):
                z.close()
                tmp.unlink(missing_ok=True)
                raise ValueError("actions.json is only for pictures recorded from a window; this run's pictures "
                                 "were imported, so its Pigeon answers with key names but cannot be played")
            # Never pack one every reader would skip (spec §13.2): the Pigeon
            # would load and never play, with only a warning to say why.
            from .runner import actions_problem
            problem = actions_problem((run_dir / "actions.json").read_bytes(), manifest)
            if problem:
                z.close()
                tmp.unlink(missing_ok=True)
                raise ValueError(f"actions.json would be skipped by every reader, so the Pigeon could never play: {problem}")
            z.write(run_dir / "actions.json", "actions.json")
        # Not for a grounded Pigeon: `grounded` wins over a label set
        # (manifest.py::detect_kind) and the manifest's output_contract is
        # `text`, so a labels.txt would sit in the file implying a label
        # contract the manifest denies. Found by packing a real 210-section
        # book onto a run that happened to have labels.
        if labels and manifest["kind"] != "grounded":
            z.writestr("labels.txt", "\n".join(labels) + "\n")
        if facts.get("skill"):
            # Data, never code. An app renders by concatenating fragments; a
            # .pigeon that shipped an evaluator would have thrown away the
            # validate-then-load doctrine the whole format rests on.
            z.writestr("skill/skill.json",
                       json.dumps(facts["skill"]["payload"], indent=1) + "\n")
        if facts.get("material_dir"):
            # The material the Pigeon answers FROM travels with it; the rows it
            # was TRAINED on still never leave (test_the_training_data_never_leaves
            # is unchanged and stays that way). chunks.json carries the section
            # text, so the source files are not packed a second time — the
            # retriever reads exactly the bytes that shipped.
            z.write(Path(facts["material_dir"]) / "chunks.json", "material/chunks.json")
            # The sections' embeddings, when the search is hybrid-v1: the
            # reader embeds only the question.
            if (manifest.get("material") or {}).get("retrieval") == "hybrid-v1":
                z.write(Path(facts["material_dir"]) / "embeddings.json", "material/embeddings.json")
        z.writestr("icon.png", icon)
        z.writestr("README.md", readme.rstrip() + "\n")
        # No flight, no report: a lesson checked by asking has no scorecard,
        # and a report.json with nothing in it would read as one.
        if facts.get("flight"):
            z.writestr("report.json", json.dumps(build_report(facts), indent=2) + "\n")
        for i, ex in enumerate(examples, 1):
            stem = f"examples/{i:03d}"
            if facts["vision"] and ex.get("image") and Path(ex["image"]).exists():
                src = Path(ex["image"])
                z.write(src, f"{stem}{src.suffix.lower()}")
                z.writestr(f"{stem}.prompt.txt", ex["question"] + "\n")
            else:
                z.writestr(f"{stem}.txt", ex["question"] + "\n")
            z.writestr(f"{stem}.expected.txt", str(ex["answer"]) + "\n")
    shutil.move(str(tmp), str(out_path))
    return manifest


def drop_stock_projector(src: Path | str, out_path: Path | str, registry: dict | None) -> dict:
    """Rewrite a packed picture Pigeon without its stock projector — the
    lineup's repack once the app holds a sealed registry (§13.1, D73). Every
    other entry is copied byte for byte, in order, with its date and
    compression; nothing is retrained or rebuilt. The rule is pack_run's own
    (`registry.stock_projector`), so a rewrite and a fresh pack agree.

    Returns {"dropped": bool, "reason"?}. Nothing is written when there is
    nothing to drop. A sealed file is refused: its seal covers the projector,
    so dropping it would break the seal."""
    src, out_path = Path(src), Path(out_path)
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        manifest = json.loads(z.read("pigeon.json"))
        if "projector.gguf" not in names:
            return {"dropped": False, "reason": "it carries no projector"}
        if "seal.json" in names:
            raise ValueError(f"{src.name} is sealed, and its seal covers the projector; seal it again after a fresh pack")
        if not registry_mod.stock_projector(registry, manifest.get("base_model") or {}):
            return {"dropped": False, "reason": "its projector is not one the registry lists for its base, so it is carried"}
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with zipfile.ZipFile(tmp, "w") as out:
            for info in z.infolist():
                if info.filename == "projector.gguf":
                    continue
                copy = zipfile.ZipInfo(info.filename, info.date_time)
                copy.compress_type = info.compress_type
                copy.external_attr = info.external_attr
                out.writestr(copy, z.read(info))
    shutil.move(str(tmp), str(out_path))
    return {"dropped": True}


# pigeon.json, icon.png, README.md and report.json, with the ZIP's own
# headers: measured at 15–25 KB on the lineup's picture Pigeons.
CARD_BYTES = 20_000


def base_display_name(file: str) -> str:
    """What a page calls a base, from its GGUF's file name:
    `Qwen3VL-2B-Instruct-Q4_K_M.gguf` is Qwen3-VL-2B, and
    `Qwen2.5-1.5B-Instruct-F16.gguf` is Qwen2.5-1.5B."""
    import re
    stem = re.sub(r"\.gguf$", "", file, flags=re.I)
    stem = re.sub(r"-(UD-)?(I?Q\d\w*|BF16|F16|F32)$", "", stem, flags=re.I)
    stem = re.sub(r"-Instruct(-\d+)?$", "", stem, flags=re.I)
    return re.sub(r"(\d)VL\b", r"\1-VL", stem)


def file_parts(run_dir: Path | str, *, registry: dict | None = None, hash_cache=None,
               example_seed: int = 11, example_count: int = 4) -> dict:
    """What pack_run would put in the file, in bytes, without packing it — the
    Send page's "What goes in the file". The projector row follows pack_run's
    own rule (`registry.stock_projector`), so the page and the file never
    disagree about whether it is carried. `base` is the base the run trained
    against, read from the run, never a default."""
    from .hashes import HashCache
    run_dir = Path(run_dir)
    manifest, facts = build_manifest(run_dir, check_hash=False)
    size = lambda p: Path(p).stat().st_size if p and Path(p).is_file() else 0  # noqa: E731
    projector, carried = facts.get("projector"), False
    if projector:
        cache = HashCache(hash_cache)
        manifest["base_model"]["mmproj"]["sha256"] = cache.hash(Path(projector))
        cache.save()
        carried = not registry_mod.stock_projector(registry, manifest["base_model"])
    labels = facts["labels"]
    examples = 0
    for i, ex in enumerate(pick_examples(example_pool(run_dir, facts), n=example_count, seed=example_seed), 1):
        if facts["vision"] and ex.get("image"):
            examples += size(ex["image"])
        examples += len((ex["question"] + "\n").encode()) + len((str(ex["answer"]) + "\n").encode())
    base_path = Path(facts["base_path"])
    out = {
        "adapter": size(facts["state"].get("adapter")),
        "projector": size(projector) if projector else None,
        "projector_carried": carried,
        "labels": len(("\n".join(labels) + "\n").encode()) if labels and manifest["kind"] != "grounded" else 0,
        "card": CARD_BYTES + examples,
        "base": {"name": base_display_name(base_path.name), "file": base_path.name, "bytes": size(base_path)},
    }
    out["total"] = out["adapter"] + (out["projector"] or 0) * carried + out["labels"] + out["card"]
    return out
