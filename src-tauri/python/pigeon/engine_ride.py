"""The check ride through the engine: Pigeon's own `pigeon-ride`, so the score
on the card is the score `pigeon-engine` gives any reader (DEV-6).

Flight School no longer asks llama-server for a ride. It writes the rows in
`pigeon-ride`'s holdout shape, runs the pinned binary on the packed file, and
reads every answer event back. Grading stays Flight School's: the same rules
as before, applied to the engine's own reading of each answer
(`pigeon_core::ask::answer`).

The control (adapter off) is a base Pigeon of the same file (D83): the same
manifest with `base_only` and no adapter, so the engine builds the same prompt
and loads no adapter. Every row's first ask must carry the same
`prompt_tokens` in both rides, or the control is refused rather than
reported. (Later turns differ by design: each ride sends its own replies.)

A row with earlier turns (`as_sent`, §5.6) is asked a turn at a time: each
round sends the replies the engine actually gave in the rounds before, as
`history` (E25), never ideal ones.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

PIN_FILE = Path(__file__).with_name("ENGINE_PIN.json")
APP = Path.home() / "Library/Application Support/mode.airoplane.flightschool"
CONTROL_SUFFIX = ".control.pigeon"


class EngineRideError(RuntimeError):
    pass


def pin() -> dict:
    """{"commit", "version", "prefix_cache", "built_from"}: the engine every ride
    is measured on. `version` is the stamp at that setting (E26), and a ride
    whose engine reports any other stamp, "prefix-cache off" included, is refused."""
    return json.loads(PIN_FILE.read_text())


def binary() -> Path:
    """`PIGEON_RIDE`, else the clone this app builds at the pinned commit."""
    env = os.environ.get("PIGEON_RIDE")
    p = Path(env) if env else APP / "engine/pigeon/src-tauri/target/release/pigeon-ride"
    if not p.is_file():
        raise EngineRideError(f"pigeon-ride is not built at {p}: build it at the commit in {PIN_FILE.name}")
    return p


def holdout_row(row: dict, history: list[dict] | None = None, question: str | None = None) -> dict:
    """A pack-table row in `pigeon-ride`'s shape: `{question, answer, image?, values?, history?, sections?}`."""
    exp = row.get("expected")
    out = {"question": row["input"] if question is None else question,
           "answer": exp if exp is None or isinstance(exp, str) else json.dumps(exp, separators=(",", ":"))}
    if row.get("image"):
        out["image"] = str(Path(row["image"]).resolve())
    if row.get("host_values"):
        out["values"] = row["host_values"]
    if history:
        out["history"] = history
    if row.get("sections"):
        out["sections"] = row["sections"]
    return out


def control_file(src: Path | str, dst: Path | str) -> Path:
    """The D83 control: everything the file carries but its adapter, with
    `base_only: true`. Not a Pigeon anyone receives; it lives beside the ride."""
    src, dst = Path(src), Path(dst)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            if info.filename == "adapter.lora.gguf":
                continue
            data = zin.read(info.filename)
            if info.filename == "pigeon.json":
                m = json.loads(data)
                m["base_only"] = True
                data = (json.dumps(m, indent=2, ensure_ascii=False) + "\n").encode()
            zout.writestr(info.filename, data)
    return dst


def run_binary(pigeon: Path, rows: list[dict], work: Path, *, models: Path, context: int = 8192,
               hash_cache: Path | None = None, exe: Path | None = None) -> dict:
    """One `pigeon-ride` over `rows`, in order. Returns {"events", "runtime", "seconds", "load_seconds"}."""
    work.mkdir(parents=True, exist_ok=True)
    holdout, out = work / "holdout.jsonl", work / "ride.json"
    holdout.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    out.unlink(missing_ok=True)
    # E26: the prefix-cache setting is part of the stamp (D98), so it is set
    # explicitly from the pin rather than left to the binary's default.
    cmd = [str(exe or binary()), "--models", str(models), "--context", str(context),
           "--prefix-cache", str(pin()["prefix_cache"]), "--out", str(out)]
    if hash_cache:
        cmd += ["--hash-cache", str(hash_cache)]
    cmd.append(f"{pigeon}={holdout}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (work / "ride.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not out.is_file():
        raise EngineRideError(f"pigeon-ride failed on {pigeon.name}: {(proc.stderr or proc.stdout)[-600:]}")
    got = json.loads(out.read_text())
    events = [r["event"] for ride in got["rides"] for r in ride["rows"]]
    if len(events) != len(rows):
        raise EngineRideError(f"pigeon-ride answered {len(events)} of {len(rows)} rows")
    return {"events": events, "runtime": got["runtime"], "seconds": got["seconds"], "load_seconds": got["load_seconds"]}


def ask(pigeon: Path | str, rows: list[dict], work: Path | str, **kw) -> dict:
    """Every row asked through the engine, earlier turns first (§5.6).
    Returns {"events" (one per row, its final ask), "runtime", "seconds", "asks"}."""
    pigeon, work = Path(pigeon), Path(work)
    histories: list[list[dict]] = [[] for _ in rows]
    first: list[dict | None] = [None] * len(rows)
    rounds = max((len(r.get("turns") or []) for r in rows), default=0)
    seconds, asks = 0.0, 0
    for k in range(rounds):
        idx = [i for i, r in enumerate(rows) if len(r.get("turns") or []) > k]
        got = run_binary(pigeon, [holdout_row(rows[i], histories[i], question=rows[i]["turns"][k]) for i in idx],
                         work / f"turn-{k + 1}", **kw)
        for i, e in zip(idx, got["events"]):
            histories[i].append({"user": rows[i]["turns"][k], "pigeon": said(e)})
            first[i] = first[i] or e
        seconds, asks = seconds + got["seconds"], asks + len(idx)
    got = run_binary(pigeon, [holdout_row(r, histories[i]) for i, r in enumerate(rows)], work / "rows", **kw)
    # Each row's first ask: the one ask a ride and its control send alike. Under
    # as_sent every later ask carries that ride's own earlier replies (§5.6).
    first = [f or e for f, e in zip(first, got["events"])]
    return {"events": got["events"], "first_events": first, "runtime": got["runtime"],
            "seconds": seconds + got["seconds"], "asks": asks + len(rows), "load_seconds": got["load_seconds"]}


def said(event: dict) -> str:
    """What the engine answered, as text (pigeon-ride's own `said`)."""
    if event.get("type") == "error":
        return ""
    return str(event.get("raw") or "").strip()


def check_pin(runtime: str) -> None:
    want = pin()["version"]
    if runtime != want:
        raise EngineRideError(f"the engine says {runtime!r}, not the pinned {want!r}: rebuild pigeon-ride at "
                              f"{pin()['commit']}")


def same_prompts(adapter_events: list[dict], control_events: list[dict]) -> list[int]:
    """Rows whose prompt differs between the ride and its control (by token count)."""
    return [i for i, (a, c) in enumerate(zip(adapter_events, control_events))
            if (a.get("stats") or {}).get("prompt_tokens") != (c.get("stats") or {}).get("prompt_tokens")]


def ride_with_control(pigeon: Path | str, rows: list[dict], work: Path | str, *, control: bool = True, **kw) -> dict:
    """The ride and, for a trained file, its control, never interleaved.
    Returns {"adapter": ask(), "control": ask() or None, "engine": version}."""
    pigeon, work = Path(pigeon), Path(work)
    if work.exists():
        shutil.rmtree(work)
    adapter = ask(pigeon, rows, work / "adapter", **kw)
    check_pin(adapter["runtime"])
    trained = "adapter.lora.gguf" in zipfile.ZipFile(pigeon).namelist()
    ctl = None
    if control and trained:
        cfile = control_file(pigeon, work / (pigeon.stem + CONTROL_SUFFIX))
        ctl = ask(cfile, rows, work / "control", **kw)
        differ = same_prompts(adapter["first_events"], ctl["first_events"])
        if differ:
            raise EngineRideError(f"the control's prompt differs from the ride's on {len(differ)} rows "
                                  f"(first: row {differ[0] + 1}), so it is not the same ask without the adapter")
    return {"adapter": adapter, "control": ctl, "engine": adapter["runtime"]}


def models_dir(base_gguf: Path | str) -> Path:
    """The folder `pigeon-ride --models` reads: the base's own, projectors in `projectors/`."""
    return Path(base_gguf).parent


def default_hash_cache() -> Path:
    return APP / "engine-hashes.json"
