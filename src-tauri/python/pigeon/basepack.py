"""A base Pigeon (spec D83): a base model with a system prompt, packed with no
training. Everything else about its kind holds — base identity, template,
temperature, `prompt_check`, card, the maker's test — so the check ride here
is a Runner too, and its rows are ones the maker supplies.

A base table is a folder:

    base-table.json     what the Pigeon is (below)
    rows.jsonl          the maker's held-out rows: {"input", "expected"?,
                        "image"?, "turns"?, "host_values"?}
    config.toml         the engine to ride with (quickbuild.rs::write_config)

`base-table.json`:

    {"name", "id", "kind", "base_gguf", "mmproj"?, "system_prompt"?,
     "prompt_template", "temperature"?, "history"?, "shots"?, "labels"?,
     "output_contract", "card"?, "rows_source", "check_row"?,
     "adapter"?, "trained_from"?, "trained_where"?, "train_python"?, "hf_home"?}

With `adapter` the same packer carries a TRAINED Pigeon whose run happened
elsewhere (Patch's router and tidier, trained with a system prompt, D85):
the file is not `base_only`, the ride is flown twice — with the adapter and
with it switched off, the control — and `prompt_check` is rendered by the
training code path (promptcheck.py, in `train_python`), as §8.6 requires of
anything trained.

`trained_where` (D97, with `adapter`) says where and how the adapter was trained,
in words ("RunPod RTX 4090, torch + peft, remote/train_remote.py at …"); it
goes into `check_ride` word for word beside `rows_source`, its sibling.

`engine` (D98) is set by the ride, never the table: the stamp of the
`pigeon-engine` that answered every row (`pigeon_engine::version()`, the
llama.cpp commit and spec version), so the score on the card names the
program that produced it.

`rows_source` says which bench and which rows the ride came from: it goes into
report.json word for word, because a score without its rows' provenance is
not one anybody can repeat.

What a ride grades: a label answer, against `expected`; a JSON answer, on the
fields `expected` names, when the row gives one — by path, so a planner row can
check `steps[].skill`, the skills it chose in order, and not inputs the
message never stated. A text answer is recorded as a sample with
no score (§12.2) — so is a JSON row without `expected`.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .gguf import base_identity, quant_name, read_header
from .manifest import FORMAT_VERSION, card_problems, schema_field, slugify, with_values

TABLE_FILE = "base-table.json"
ROWS_FILE = "rows.jsonl"
KINDS = ("classifier", "router", "extractor", "freeform", "persona")
EXAMPLES = 4


def load_table(table_dir: Path | str) -> tuple[dict, list[dict]]:
    """The table and its rows, or ValueError naming what is wrong."""
    d = Path(table_dir)
    t = json.loads((d / TABLE_FILE).read_text())
    for k in ("name", "kind", "base_gguf", "prompt_template", "output_contract", "rows_source"):
        if not t.get(k):
            raise ValueError(f"{TABLE_FILE} has no {k}")
    if t["kind"] not in KINDS:
        raise ValueError(f"a base Pigeon is one of {', '.join(KINDS)}, not {t['kind']!r}")
    if t["kind"] == "persona" and not t.get("system_prompt"):
        raise ValueError("a persona carries its system prompt")
    if t["kind"] in ("classifier", "router") and not t.get("labels"):
        raise ValueError(f"a {t['kind']} needs its labels")
    rows = [json.loads(line) for line in (d / ROWS_FILE).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{ROWS_FILE} has no rows: the maker's test needs at least one")
    for i, r in enumerate(rows, 1):
        if not isinstance(r.get("input"), str):
            raise ValueError(f"row {i} has no input")
        if r.get("image") and not Path(r["image"]).is_file():
            raise ValueError(f"row {i}'s picture is missing: {r['image']}")
    return t, rows


def base_model(gguf: Path, mmproj: Path | None, *, registry: dict | None, hashes) -> dict:
    """`base_model` read from the base's own header, never typed."""
    header = read_header(gguf)
    ident = base_identity(header)
    bm = {"file": gguf.name, "sha256": hashes.hash(gguf) if hashes else None,
          "arch": ident["arch"], "name": ident["name"], "size": ident["size"],
          **({"finetune": ident["finetune"]} if ident["finetune"] else {}),
          "quant": quant_name(header), "vision": mmproj is not None}
    if registry:
        from .registry import entry_for
        entry = entry_for(registry, ident)
        if entry and entry.get("id"):
            bm["registry_id"] = entry["id"]
    if mmproj is not None:
        bm["mmproj"] = {"file": mmproj.name, "sha256": hashes.hash(mmproj) if hashes else None}
    return {k: v for k, v in bm.items() if v is not None}


def manifest_for(t: dict, bm: dict, *, revision: int) -> dict:
    pictures = "{image}" in t["prompt_template"]
    m = {"format_version": FORMAT_VERSION, "name": t["name"], "id": t.get("id") or slugify(t["name"]),
         "kind": t["kind"], "base_only": None if t.get("adapter") else True, "base_model": bm,
         "trained_from": t.get("trained_from"), "prompt_template": t["prompt_template"],
         "temperature": t.get("temperature", 0), "inputs": ["image", "text"] if pictures else ["text"],
         "output_contract": t["output_contract"], "trained_on": None, "trained_with": "Flight School",
         "author": t.get("author"), "license": t.get("license", "Apache-2.0"), "signature": ""}
    if t.get("history"):
        m["history"] = t["history"]
    card = dict(t.get("card") or {})
    if card:
        card["revision"] = int(revision)
        m["card"] = card
    return {k: v for k, v in m.items() if v is not None}


def _content(text: str, image: str | None):
    if not image:
        return text
    from .picturesize import _data_uri
    return [{"type": "image_url", "image_url": {"url": _data_uri(Path(image))}}, {"type": "text", "text": text}]


def ask_kwargs(m: dict, labels: list[str] | None, values: dict | None) -> dict:
    """§8.4: how a Runner decodes this Pigeon for one ask — the file's
    temperature, its max_tokens, and its constraint with the host's lists."""
    oc = m["output_contract"]
    kw = {"max_tokens": oc["max_tokens"], "temperature": m.get("temperature", 0) if oc["type"] == "text" else 0}
    if not oc.get("constrained"):
        return kw
    values = values or {}
    if oc["type"] == "label":
        from instructor.fly import label_grammar
        kw["grammar"] = label_grammar(values.get("label") or labels or [])
    elif oc["type"] == "json" and oc.get("output_schema"):
        held = with_values(oc["output_schema"], values)
        if held is None:
            raise ValueError(f"the row's host values name a field the schema does not have: {sorted(values)}")
        kw["json_schema"] = held
    return kw


def grade(m: dict, labels: list[str] | None, row: dict, raw: str) -> tuple[bool | None, dict]:
    """(passed or None for a sample, the answer as a Runner reads it)."""
    from .runner import read_answer
    got = read_answer(m["output_contract"], raw, labels=(row.get("host_values") or {}).get("label") or labels,
                      kind=m["kind"])
    return judge(m, row, got), got


def judge(m: dict, row: dict, got: dict) -> bool | None:
    """Whether an answer, already read (§8.5), is right: None for a sample."""
    if "expected" not in row or m["output_contract"]["type"] == "text":
        return None
    if got["kind"] == "label":
        return got["value"] == row["expected"]
    if got["kind"] == "json":
        # The fields the row names, no others: a planner row checks its steps,
        # not the words of its `about`.
        want = row["expected"] if not isinstance(row["expected"], str) else json.loads(row["expected"])
        return all(field_value(got["value"], k) == v for k, v in want.items())
    return False


def field_value(obj, path: str):
    """The value at a `host_values`-style path: `a.b` steps into objects and
    `a[]` maps over a list, so `steps[].skill` is every step's skill, in order."""
    cur = [obj]
    many = False
    for seg in path.split("."):
        items = seg.endswith("[]")
        name = seg[:-2] if items else seg
        nxt = []
        for c in cur:
            v = c.get(name) if isinstance(c, dict) else None
            if items:
                nxt.extend(v if isinstance(v, list) else [])
            else:
                nxt.append(v)
        cur, many = nxt, many or items
    return cur if many else cur[0]


def ride(srv, t: dict, m: dict, rows: list[dict], *, lora=None) -> dict:
    """The maker's test, asked exactly as a Runner asks: the system prompt,
    the shots, the conversation's own earlier turns, then the row. `lora`
    is passed through per ask: [{"id": 0, "scale": 0.0}] is the control."""
    from .runner import messages
    labels, shots = t.get("labels"), t.get("shots")
    out = []
    for row in rows:
        kw = ask_kwargs(m, labels, row.get("host_values"))
        if lora is not None:
            kw["lora"] = lora
        history: list[dict] = []
        # §5.6: under as_sent the earlier replies are the ones the Pigeon
        # actually gave in this run, never ideal ones.
        for said in row.get("turns") or []:
            msgs = messages(m, said, system_prompt=t.get("system_prompt"), shots=shots, history=history)
            reply = srv.chat([{"role": x["role"], "content": x["content"]} for x in msgs], seed=11, think=False, **kw)
            history.append({"user": said, "pigeon": reply.strip()})
        msgs = messages(m, row["input"], system_prompt=t.get("system_prompt"), shots=shots, history=history)
        sent = [{"role": x["role"], "content": _content(x["content"], row.get("image") if x.get("image_first") else None)}
                for x in msgs]
        raw = srv.chat(sent, seed=11, think=False, **kw)
        passed, got = grade(m, labels, row, raw)
        out.append(_row_record(row, raw, passed, got))
    return _summary(m, out)


def _row_record(row: dict, raw: str, passed: bool | None, got: dict) -> dict:
    return {"question": row["input"], "expected": row.get("expected"), "actual": raw.strip(),
            "passed": passed, "how": "sample" if passed is None else "exact",
            **({"image": Path(row["image"]).name} if row.get("image") else {}),
            **({"turns": len(row["turns"])} if row.get("turns") else {}),
            **({"off_contract": True} if got["kind"] == "off_contract" else {}),
            # A text row with an exact target (the tidier's) says whether
            # it came back identical. Not a score: prose is a sample (§12.2).
            **({"same_as_expected": raw.strip() == str(row["expected"]).strip()}
               if passed is None and "expected" in row else {})}


def _summary(m: dict, out: list[dict]) -> dict:
    graded = [r for r in out if r["passed"] is not None]
    return {"runtime": "llama.cpp", "rows": out, "temperature": m.get("temperature", 0)
            if m["output_contract"]["type"] == "text" else 0,
            "constrained": bool(m["output_contract"].get("constrained")),
            "passed": sum(r["passed"] for r in graded) if graded else None,
            "total": len(graded) if graded else None, "samples": len(out) - len(graded),
            "off_contract": sum(1 for r in out if r.get("off_contract")),
            "same_as_expected": sum(1 for r in out if r.get("same_as_expected")) if any("same_as_expected" in r for r in out) else None}


def engine_flight(m: dict, rows: list[dict], events: list[dict]) -> dict:
    """A ride's rows and score from the engine's answer events: the engine read
    each answer (`pigeon_core::ask::answer`); Flight School grades it."""
    from .engine_ride import said
    out = []
    for row, e in zip(rows, events):
        if e.get("type") == "error":
            got = {"kind": "off_contract", "raw": ""}
        else:
            got = e.get("answer") or {"kind": "off_contract", "raw": said(e)}
        rec = _row_record(row, said(e), judge(m, row, got), got)
        rec["ms"] = (e.get("stats") or {}).get("total_ms")
        if e.get("type") == "error":
            rec["error"] = e.get("message") or e.get("code") or "error"
        out.append(rec)
    return _summary(m, out)


def picture_markup(template: str) -> str:
    """What this base's chat template writes for one picture: the template
    rendered with a picture and without, and the difference (§8.6)."""
    from .promptproof import render_chat
    a = render_chat(template, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}])
    b = render_chat(template, [{"role": "user", "content": "x"}])
    i = 0
    while i < len(b) and a[i] == b[i]:
        i += 1
    j = 0
    while j < len(b) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return a[i:len(a) - j]


def trained_prompt_check(t: dict, row: dict) -> dict:
    """§8.6 for a trained Pigeon: rendered by the code that built its training
    rows (promptcheck.py, mlx-lm's tokenizer), in the training venv."""
    import subprocess
    from .runner import fill
    # The user turn as training rows carried it: the input filled into the
    # template (§8.2), not the bare input. A router's "{input}" hid the
    # difference; the asks check's template showed it (byte 4,275).
    req = {"mode": "text", "repo": t["trained_from"]["repo"], "hf_home": t.get("hf_home"),
           "user": fill(t["prompt_template"], row["input"]),
           "system": t.get("system_prompt"), "shots": t.get("shots")}
    out = subprocess.run([t["train_python"], str(Path(__file__).with_name("promptcheck.py")), json.dumps(req)],
                         capture_output=True, text=True, timeout=600)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    if "canonical" not in got:
        raise ValueError(f"the training prompt could not be rendered: {got.get('error')}")
    return {"input": row["input"].strip(), "canonical": got["canonical"]}


def build_prompt_check(m: dict, t: dict, row: dict, template: str, picture_tokens: int | None) -> dict:
    """D83: a base Pigeon has no training rows, so its maker renders the
    template itself — the base's own chat template, as every Runner does."""
    from .promptproof import canonical
    pic = {"markup": picture_markup(template), "tokens": picture_tokens} if row.get("image") else None
    text = canonical(m, row["input"], template, system_prompt=t.get("system_prompt"), shots=t.get("shots"),
                     picture=pic)
    pc_input = {"text": row["input"].strip(), "image": "examples/check" + Path(row["image"]).suffix.lower()} \
        if row.get("image") else row["input"].strip()
    return {"input": pc_input, "canonical": text}


def report(t: dict, m: dict, flight: dict, rows: list[dict]) -> dict:
    ride_ = {k: flight[k] for k in ("passed", "total", "samples", "off_contract", "runtime", "temperature", "constrained",
                                    "control_passed", "same_as_expected")}
    return {"check_ride": {**ride_, "file_sha256": m["base_model"].get("sha256"), "passes": 1,
                           "history": (m.get("history") or {}).get("shape", "none"),
                           "rows_source": t["rows_source"],
                           # D97 is `trained_where` (spec Draft 11): the manifest's `trained_on` is a
                           # date. A table written before the rename still says `trained_on`.
                           **({"trained_where": t.get("trained_where") or t["trained_on"]}
                              if t.get("trained_where") or t.get("trained_on") else {}),
                           **({"engine": flight["engine"]} if flight.get("engine") else {}),
                           "rows": flight["rows"]},
            "check_rides": [{"runtime": "llama.cpp", "passed": flight["passed"], "total": flight["total"],
                             "control_passed": flight["control_passed"], "temperature": ride_["temperature"],
                             "constrained": ride_["constrained"], "file_sha256": m["base_model"].get("sha256"),
                             **({"engine": flight["engine"]} if flight.get("engine") else {})}],
            "practice_set": None,
            "notes": [(f"Trained elsewhere; packed by Flight School from {Path(t['adapter']).name}." if t.get("adapter")
                       else f"A base Pigeon (D83): {m['base_model']['file']} with no adapter.") + f" Its test rows: {t['rows_source']}"]}


def readme(t: dict, m: dict, flight: dict) -> str:
    score = (f"{flight['passed']} of {flight['total']} right on its check ride (higher is better)"
             if flight["total"] else f"{flight['samples']} answers recorded as samples, with no score")
    return (f"# {m['name']}\n\n{(m.get('card') or {}).get('does', '')}\n\n"
            f"A base Pigeon: {m['base_model']['file']} with no trained weights, asked with "
            f"{'its system prompt' if t.get('system_prompt') else 'no system prompt'}"
            f"{' and ' + str(len(t['shots'])) + ' fixed example turns' if t.get('shots') else ''}.\n\n"
            f"Check ride: {score}. Rows: {t['rows_source']}.\n")


def pack_base(table_dir: Path | str, out_path: Path | str, *, cfg, registry: dict | None = None, hashes=None,
              revision: int = 1, icon_png: bytes | None = None, server=None, rider=None) -> dict:
    """Write `<out_path>`: the manifest, its system prompt and shots, its
    adapter when the table names one, the ride's report and examples.

    The ride runs through the engine (DEV-6): the file is packed without a
    report, ridden by `pigeon-ride` with its D83 control, then written again
    with the report. `rider(pigeon_path, rows)` returns what
    `engine_ride.ride_with_control` returns, injectable so tests need no model.
    `server(spec)` opens llama-server for one thing only, counting a check
    picture's tokens (D80); no answer is asked of it."""
    from .promptproof import check as proof_check
    table_dir, out_path = Path(table_dir), Path(out_path)
    t, rows = load_table(table_dir)
    gguf, mmproj = Path(t["base_gguf"]), Path(t["mmproj"]) if t.get("mmproj") else None
    if "{image}" in t["prompt_template"] and mmproj is None:
        raise ValueError("a Pigeon that looks needs its base's projector (mmproj)")
    bm = base_model(gguf, mmproj, registry=registry, hashes=hashes)
    m = manifest_for(t, bm, revision=revision)
    template = read_header(gguf).get("tokenizer.chat_template")
    if not isinstance(template, str):
        raise ValueError(f"{gguf.name} carries no chat template to render the prompt with")
    for f in (m["output_contract"].get("host_values") or []):
        if m["output_contract"]["type"] == "json" and schema_field(m["output_contract"].get("output_schema"), f) is None:
            raise ValueError(f"host_values names {f}, which output_schema does not have")

    check_row = rows[int(t.get("check_row", 0))]
    adapter = Path(t["adapter"]) if t.get("adapter") else None
    tokens = None
    if check_row.get("image"):
        from instructor.engine import ServerSpec
        from .picturesize import count_in_llama
        if server is None:
            from instructor.engine import LlamaServer
            server = lambda spec: LlamaServer(cfg, spec, log=table_dir / "count-server.log")  # noqa: E731
        with server(ServerSpec(model=gguf, mmproj=mmproj, ctx=8192)) as srv:
            tokens = count_in_llama(srv, check_row["image"], check_row["input"].strip())
    m["prompt_check"] = trained_prompt_check(t, check_row) if adapter is not None \
        else build_prompt_check(m, t, check_row, template, tokens)
    # The Runner's half, before anything is written: the prompt a Runner
    # builds from this file must be the one written down.
    proof = proof_check(m, template, system_prompt=t.get("system_prompt"), shots=t.get("shots"),
                        picture={"markup": picture_markup(template), "tokens": tokens} if tokens else None)
    if not proof["ok"]:
        raise ValueError(f"prompt_check does not reproduce (byte {proof['position']})")
    if icon_png is None:
        from pigeonlook import random_look, render_png
        icon_png = render_png(random_look(m["name"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    riding = out_path.with_suffix(".riding.pigeon")
    _write_pigeon(riding, t, m, adapter, check_row, icon_png)
    if rider is None:
        from . import engine_ride
        rider = lambda path, rs: engine_ride.ride_with_control(  # noqa: E731
            path, rs, table_dir / "engine-ride", models=engine_ride.models_dir(gguf),
            hash_cache=engine_ride.default_hash_cache())
    try:
        got = rider(riding, rows)
    finally:
        riding.unlink(missing_ok=True)
    flight = engine_flight(m, rows, got["adapter"]["events"])
    flight["engine"] = got["engine"]
    flight["control_passed"] = engine_flight(m, rows, got["control"]["events"])["passed"] if got.get("control") else None
    if m.get("card") and flight["total"]:
        m["card"]["check"] = {"passed": flight["passed"], "total": flight["total"]}
    if m.get("card") and card_problems(m["card"]):
        raise ValueError("the card cannot be packed: " + "; ".join(card_problems(m["card"])))
    _write_pigeon(out_path, t, m, adapter, check_row, icon_png, flight=flight, rows=rows)
    return {"manifest": m, "flight": flight}


def _write_pigeon(out_path: Path, t: dict, m: dict, adapter: Path | None, check_row: dict, icon_png: bytes, *,
                  flight: dict | None = None, rows: list[dict] | None = None) -> None:
    """The file. Without a flight it carries no report, README or examples:
    that is the copy the engine rides, never one anybody receives."""
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("pigeon.json", json.dumps(m, indent=2, ensure_ascii=False) + "\n")
        if adapter is not None:
            z.write(adapter, "adapter.lora.gguf")
        if t.get("system_prompt"):
            z.writestr("system_prompt.txt", t["system_prompt"])
        if t.get("shots"):
            z.writestr("shots.json", json.dumps(t["shots"], indent=1, ensure_ascii=False) + "\n")
        if t.get("labels"):
            z.writestr("labels.txt", "\n".join(t["labels"]) + "\n")
        if check_row.get("image"):
            z.write(check_row["image"], m["prompt_check"]["input"]["image"])
        z.writestr("icon.png", icon_png)
        if flight is not None:
            z.writestr("README.md", readme(t, m, flight))
            z.writestr("report.json", json.dumps(report(t, m, flight, rows), indent=2, ensure_ascii=False) + "\n")
            for i, (row, r) in enumerate(zip(rows[:EXAMPLES], flight["rows"]), 1):
                if row.get("image") or row.get("turns"):
                    continue
                z.writestr(f"examples/{i:03d}.txt", row["input"] + "\n")
                exp = row.get("expected", r["actual"])
                z.writestr(f"examples/{i:03d}.expected.txt",
                           (exp if isinstance(exp, str) else json.dumps(exp, separators=(",", ":"))) + "\n")
    shutil.move(str(tmp), str(out_path))
