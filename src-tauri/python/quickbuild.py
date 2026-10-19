#!/usr/bin/env python3
"""Flight School's training driver — the vendored Instructor pipeline's
train-and-check-ride stages, driven with machine-readable progress.

Run by Rust (quickbuild.rs) through the training venv's python3, never by a
person. Every interesting moment prints one line starting with `@@QB ` followed
by JSON — pyproc forwards every line to the log console, and a Rust LineWatcher
lifts the @@QB ones into structured `qb-event`s for the UI. Everything else
this process prints is honest logging and stays in the console.

One subcommand (it takes --config and --run-dir):

  table    the rows and the exam are the person's, written by Flight School
           from the table → measure → dataset → train → convert → check ride.

The one-click brief→cartridge flow this file used to drive (ask, plan, revise,
exam, build, run, import, rebuild, retrain, fly, aim) is gone with its screens;
Teach is the one way in. The pipeline itself is the vendored `instructor`
package, untouched — this file is orchestration only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from instructor import builder, card as card_mod, config as config_mod  # noqa: E402
from instructor import convert, dataset, fly as fly_mod, grade as grade_mod, planspec  # noqa: E402
from instructor import preflight, train_text, train_vision  # noqa: E402
from instructor.engine import engine_lock  # noqa: E402
from instructor.generate import Row  # noqa: E402
from instructor.planspec import PRESETS  # noqa: E402


def emit(kind: str, **data) -> None:
    print("@@QB " + json.dumps({"kind": kind, **data}), flush=True)


def set_state(run_dir: Path, **patch) -> dict:
    path = run_dir / "state.json"
    state = {}
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
    state.update(patch)
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


class Clock:
    """Wall time per stage, so a change meant to make a run faster has a
    measured before and after rather than an impression.

    Every stage lands three ways: a `timing` event (the log console shows it
    as it happens), `timings.json` beside the run, and the facts that decide
    the time — RAM, slots, checkpointing, window — in the same file, because a
    duration compared across machines without them compares nothing.
    """

    def __init__(self, run_dir: Path):
        self.path = run_dir / "timings.json"
        self.started = time.monotonic()
        self.stages: dict[str, float] = {}
        self.facts: dict = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - t0)

    def record(self, name: str, seconds: float) -> None:
        self.stages[name] = round(seconds, 2)
        emit("timing", stage=name, seconds=self.stages[name])
        self.save()

    def note(self, **facts) -> None:
        self.facts.update(facts)
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "total": round(time.monotonic() - self.started, 2),
            "stages": self.stages, "facts": self.facts}, indent=2) + "\n")


def hooks_for(run_dir: Path) -> builder.Hooks:
    return builder.Hooks(
        phase=lambda title, detail: emit("phase", title=title, detail=detail),
        note=lambda msg: emit("note", msg=msg),
        warn=lambda msg: emit("warn", msg=msg),
        question_progress=lambda group, i, n: emit("questions", group=group, i=i, n=n),
        answer_progress=lambda i, n: emit("answers", i=i, n=n),
        answer_sample=lambda group, q, a: emit("sample", group=group, question=q[:300], answer=a[:500]),
        train_progress=lambda i, total, kind, loss: emit(
            "train", iter=i, total=total, phase=kind, loss=loss),
    )


# ---------------------------------------------------------------------------
# stages


def stage_table(cfg, args, run_dir: Path, plan: dict, clock: Clock):
    """Train from a two-column table (Milestone 5): the rows and the exam are
    the person's, written by Flight School from the table — `rows.jsonl` holds
    the rows that train, `exam.jsonl` the exam, which is pinned verbatim so
    every round is graded on the same questions. Text or pictures: a vision
    table trains through mlx-vlm with its images, a text table through mlx-lm.
    Then convert, and the check ride."""
    set_state(run_dir, stage="build", error=None)
    emit("stage", stage="build")
    base = json.loads((run_dir / "base.json").read_text())
    student_repo, student_gguf = base["student_repo"], Path(base["student_gguf"])
    vision = plan.get("modality") == "vision"
    def read(name):
        return [Row(question=d["question"], answer=d["answer"], group=d.get("group", ""), image=d.get("image"))
                for d in (json.loads(l) for l in (run_dir / name).read_text().splitlines() if l.strip())]
    train_rows, exam_rows = read("rows.jsonl"), read("exam.jsonl")
    if len(train_rows) < 10:
        emit("error", stage="build", msg=f"only {len(train_rows)} rows to train on — too few")
        set_state(run_dir, error="too few rows")
        return None
    rows = train_rows + exam_rows
    for r in rows:
        if vision and not (r.image and Path(r.image).is_file()):
            emit("error", stage="build", msg=f"a row's picture is missing: {r.image}")
            set_state(run_dir, error="a picture is missing")
            return None

    h = hooks_for(run_dir)
    h._say(h.phase, "measure", "measuring the rows through the base's own tokenizer")
    if not vision:
        with clock.stage("weights"):
            train_text.ensure_weights(cfg, student_repo, on_line=lambda l: emit("note", msg=l))
    with clock.stage("measure"):
        pf = preflight.measure(cfg, student_repo, [{"q": r.question, "a": r.answer} for r in rows],
                               extra_tokens=builder.VISION_IMAGE_TOKEN_ALLOWANCE if vision else 0)
    if pf.verdict == "wont_fit":
        if vision:
            emit("error", stage="build", msg=f"this will not fit: {pf.est_memory_gb:.1f} GB estimated "
                                             f"against a {pf.budget_gb:.1f} GB budget")
            set_state(run_dir, error="will not fit")
            return None
        dims = preflight.model_dims(cfg, student_repo)
        pf, kept, dropped, warn = preflight.fit_to_budget(pf, dims, train_rows, max(20, len(exam_rows) + 10))
        emit("warn", msg=warn)
        train_rows = kept
        rows = train_rows + exam_rows

    h._say(h.phase, "dataset", "the exam is yours, pinned; the rest trains")
    ds = dataset.write_dataset(
        rows, run_dir, holdout_count=len(exam_rows),
        answer_token_lengths=pf.answer_lengths, claimed_answer_tokens=plan["max_answer_tokens"],
        max_seq_length=pf.max_seq_length, vision=vision,
        holdout=[{"question": r.question, "image": r.image} for r in exam_rows], strict_pin=True)
    (run_dir / "measured.json").write_text(json.dumps({
        "answer_tokens_claimed": ds.claimed_answer_tokens,
        "answer_tokens_measured": ds.measured_answer_tokens,
        "max_seq_length": pf.max_seq_length}, indent=2) + "\n")

    total = PRESETS[plan["preset"]]
    clock.note(ram_gb=cfg.total_ram_gb, budget_gb=cfg.generation_budget_gb,
               modality="vision" if vision else "text", preset=plan["preset"], iters=total,
               train_rows=len(train_rows), exam_rows=len(exam_rows),
               max_seq_length=pf.max_seq_length, grad_checkpoint=pf.grad_checkpoint)
    h._say(h.phase, "train", f"{plan['preset']} preset · {total} iters")
    progress = lambda it, n, kind, loss, its=None, peak=None: emit(
        "train", iter=it, total=n, phase=kind, loss=loss, its=its, peak_gb=peak)
    salvaged = None
    speed: dict = {}
    if vision:
        with clock.stage("train"), engine_lock(cfg, "vision training (table)"):
            vres = train_vision.train(
                cfg, data_dir=ds.data_dir, adapter_dir=run_dir / "adapters", repo=student_repo,
                preset=plan["preset"], max_seq_length=pf.max_seq_length, grad_checkpoint=pf.grad_checkpoint,
                log_path=run_dir / "train.log",
                on_speed=lambda its, peak: speed.update(its=its, peak=peak),
                on_progress=lambda i, n, k, l: progress(i, n, k, l, speed.get("its"), speed.get("peak")))
        salvaged = vres.salvaged_from_iter
        if salvaged is not None:
            emit("warn", msg=f"training stopped on a trainer error; the checkpoint from iteration {salvaged} was kept")
    else:
        with clock.stage("train"), engine_lock(cfg, "training (table)"):
            res = train_text.train(
                cfg, data_dir=ds.data_dir, adapter_dir=run_dir / "adapters", repo=student_repo,
                preset=plan["preset"], max_seq_length=pf.max_seq_length, grad_checkpoint=pf.grad_checkpoint,
                log_path=run_dir / "train.log",
                on_progress=lambda p: progress(p.iter, p.total, p.kind, p.loss, p.its, p.peak_gb))
        if res.early_stopped:
            emit("warn", msg="stopped early — validation loss rose three times running; the best checkpoint was kept")
    clock.note(early_stopped=None if vision else bool(res.early_stopped), salvaged_from_iter=salvaged)

    h._say(h.phase, "convert", "mlx → PEFT → GGUF, then the verification gate")
    with clock.stage("convert"):
        conv = convert.to_gguf(cfg, adapter_dir=run_dir / "adapters", out_dir=run_dir / "export",
                               repo=student_repo, name=plan["job_name"], vision=vision)
    for n in conv.notes:
        emit("warn", msg=n)
    notes = ["Trained from a table the person built: every answer of record is theirs."]
    if salvaged is not None:
        notes.append(f"Training stopped on a trainer error; the iter-{salvaged} checkpoint was kept.")
    c = card_mod.from_plan(
        plan, base_path=student_gguf, adapter_path=conv.gguf_path,
        train_examples=ds.n_train + ds.n_valid, holdout=ds.n_holdout, iters=total, notes=notes,
        measured_answer_tokens=ds.measured_answer_tokens, claimed_answer_tokens=ds.claimed_answer_tokens)
    card_path = card_mod.write(c, run_dir / "export" / f"{plan['job_name']}.card.txt")
    import shutil
    shutil.copy2(ds.holdout_path, run_dir / "export" / "holdout.jsonl")
    set_state(run_dir, rows=len(rows),
              dataset={"train": ds.n_train, "valid": ds.n_valid, "holdout": ds.n_holdout},
              adapter=str(conv.gguf_path), card=str(card_path),
              conversion={"verified": bool(conv.verified), "worst_relative_error": conv.worst_relative_error})
    emit("built", rows=len(rows), train=ds.n_train, valid=ds.n_valid, holdout=ds.n_holdout,
         adapter=str(conv.gguf_path), salvaged=salvaged)
    return conv.gguf_path


def _reader_chunks(run_dir: Path) -> list[dict] | None:
    p = run_dir / "material" / "chunks.json"
    try:
        return json.loads(p.read_text()).get("chunks") if p.is_file() else None
    except (OSError, ValueError):
        return None


def stage_fly(cfg, run_dir: Path, plan: dict, clock: Clock):
    set_state(run_dir, stage="fly", error=None)
    emit("stage", stage="fly")
    export_dir = run_dir / "export"
    adapter = next(export_dir.glob("*.gguf"), None)
    if not adapter:
        emit("error", stage="fly", msg="no adapter in the export folder — the build didn't finish")
        set_state(run_dir, error="no adapter to fly")
        return None
    holdout = dataset.read_holdout(export_dir / "holdout.jsonl")
    budget = fly_mod.answer_budget(adapter, plan)
    # Not label_set alone: this flag drives a second pass that OVERWRITES
    # passed/how on every row, so a rules-graded run would have its
    # deterministic verdicts replaced by judged ones.
    needs_judge = grade_mod.needs_judge(plan)
    rows: list[dict] = []

    def on_row(r):
        rows.append(r.__dict__)
        emit("ride", i=len(rows), n=len(holdout), passed=r.passed,
             how=r.how, question=r.question[:120])

    control_seen: list[dict] = []

    def on_control_row(r):
        control_seen.append(r.__dict__)
        emit("control", i=len(control_seen), n=len(holdout), passed=r.passed,
             how=r.how, question=r.question[:120])

    # Two passes when a judge is needed: the cartridge answers, its server is
    # released, then the stronger teacher grades — the engine lock means they
    # never share the machine.
    # From the PLAN, not a constant: a vision cartridge's check ride has to
    # load the projector alongside the base, or every held-out question is
    # asked of a model that cannot see the image it is about.
    vision = plan.get("modality") == "vision"
    # `control=True` is not optional and is not a setting. The ride asks every
    # held-out question twice: once with the adapter and once with it switched
    # off. Without the second number a score cannot tell "the adapter learned
    # the job" from "the prompt got shorter" — measured spread on one job was
    # 46/47 with, 5/47 without.
    grounded = None
    chunks = _reader_chunks(run_dir) if plan.get("grounded") else None
    # DEV-6: the ride runs through the engine every reader serves with, on the
    # file itself: a copy is packed with no report, `pigeon-ride` asks it and
    # its D83 control (the same file without its adapter), and the copy is
    # deleted. The grading below is unchanged.
    from pigeon import engine_ride, pack as pack_mod
    ride_copy = export_dir / "ride.pigeon"
    loading = time.monotonic()
    pack_mod.pack_run(run_dir, ride_copy, ride_copy=True, llama_server=cfg.llama_server,
                      model_dirs=[Path(cfg.target_gguf).parent])
    clock.record("ride.pack_copy", time.monotonic() - loading)
    asking = time.monotonic()
    try:
        got = engine_ride.ride_with_control(ride_copy, fly_mod.engine_rows(plan, holdout, chunks),
                                            run_dir / "engine-ride", models=engine_ride.models_dir(cfg.target_gguf),
                                            hash_cache=engine_ride.default_hash_cache())
    finally:
        ride_copy.unlink(missing_ok=True)
    clock.record("ride.engine", time.monotonic() - asking)
    said = {False: [engine_ride.said(e) for e in got["adapter"]["events"]],
            True: [engine_ride.said(e) for e in got["control"]["events"]]}
    passes: dict[str, float] = {}
    flight = fly_mod.run_flight(
        None, plan, holdout,
        max_answer_tokens=budget,
        judge=False,
        control=True,
        on_row=None if needs_judge else on_row,
        on_control_row=None if needs_judge else on_control_row,
        timings=passes,
        chunks=chunks,
        answers=lambda control: said[control],
    )
    flight.engine = got["engine"]
    emit("note", msg=f"asked through {got['engine']}")
    # A Reader's second number: does it answer from the passage, or from
    # memory? Exam answers that quote a number from their own passage are
    # asked again with that number changed (instructor/grounding.py), with
    # the adapter and without. The engine asks a section by its ordinal and
    # cannot be handed a changed passage, so this one still runs on
    # llama-server.
    if plan.get("grounded"):
        from instructor import grounding as grounding_mod
        exam_path = run_dir / "exam.jsonl"
        exam_rows = [json.loads(l) for l in exam_path.read_text().splitlines() if l.strip()] if exam_path.exists() else []
        cases = grounding_mod.swap_cases(exam_rows)
        emit("note", msg=f"fact swap: {len(cases)} exam answers quote a number from their passage — asking each again with the number changed, with the adapter and without")
        with fly_mod.flight_server(cfg, adapter, vision=vision) as srv, clock.stage("ride.fact_swap"):
            grounded = grounding_mod.run_swaps(srv, cases, max_tokens=budget)
    if needs_judge:
        emit("note", msg=f"grading {len(flight.rows)} prose answers with the teacher model — a machine opinion, read the rows yourself")
        loading = time.monotonic()
        with fly_mod.judge_server(cfg, "instructor") as jsrv:
            clock.record("judge.load", time.monotonic() - loading)
            clock.note(judge_slots=jsrv.spec.n_parallel)
            grading = time.monotonic()
            def grade(row):
                try:
                    return row, fly_mod._judge(jsrv, plan, row.question,
                                               row.expected, row.actual,
                                               own_server=False)
                except Exception:
                    return row, None

            def apply(result):
                row, verdict = result
                if verdict is None:
                    row.how, row.passed = "judge-error", False
                else:
                    row.passed, row.how = verdict, "judge"
                    flight.graded_by_model += 1
                on_row(row)

            def apply_control(result):
                row, verdict = result
                if verdict is None:
                    row.how, row.passed = "judge-error", False
                else:
                    row.passed, row.how = verdict, "judge"
                on_control_row(row)

            # Every grading call is identical in adapter state (this server has
            # none loaded), so they batch cleanly. One slot means the sequential
            # loop this replaced.
            fly_mod._fan_out(grade, flight.rows, jsrv.spec.n_parallel, apply)
            # The control is graded by the SAME judge, the same way. A control
            # graded differently is not a control.
            if flight.control_rows:
                fly_mod._fan_out(grade, flight.control_rows,
                                 jsrv.spec.n_parallel, apply_control)
            clock.record("judge.grade", time.monotonic() - grading)

    # A Reader's check that needs no judge, led with (SPEC §12.1): the same
    # source sentence as the expected answer — the rule Pigeon checks a
    # grounded Pigeon's examples by. Rows get `same_source` before they are saved.
    same_source = None
    if plan.get("grounded"):
        from instructor import grounding as grounding_src
        same_source = grounding_src.source_check(flight, _reader_chunks(run_dir))
        if same_source:
            base = "" if same_source["base_sentence"] is None else f", untrained {same_source['base_sentence']} of {same_source['base_n']}"
            emit("note", msg=f"same sentence of the passage as the expected answer: {same_source['sentence']} of {same_source['n']}{base}; "
                             f"{same_source['same']} of those also cite its section")

    # The second ride, in MLX where the adapter trained (instructor/mlx_ride.py):
    # the llama.cpp number is the one hosts see; this one shows what was lost
    # between training and serving. Never fatal — the first ride stands alone.
    rides = [{"runtime": flight.runtime, **({"engine": flight.engine} if flight.engine else {}),
              "passed": flight.passed, "total": flight.total,
              "control_passed": flight.control_passed, "temperature": flight.temperature,
              "constrained": flight.constrained}]
    from instructor.mlx_ride import run_mlx_ride, serving_trails
    from pigeon.promptcheck import precision as _precision
    base = json.loads((run_dir / "base.json").read_text())
    with clock.stage("ride.mlx"):
        mlx, why = run_mlx_ride(cfg, run_dir, plan, holdout, max_answer_tokens=budget, repo=base["student_repo"],
                                chunks=_reader_chunks(run_dir) if plan.get("grounded") else None,
                                precision=_precision(base["student_repo"], str(cfg.hf_home)))
    if mlx:
        rides.append(mlx)
        emit("note", msg=f"second ride, in MLX where it trained: {mlx['passed']} of {mlx['total']}")
        lost = serving_trails(rides[0], mlx)
        if lost:
            emit("warn", msg=lost)
    else:
        emit("note", msg=f"no second ride in MLX: {why}")

    ride_path = fly_mod.save_flight(flight, export_dir / "check-ride.json")
    if grounded is not None or same_source is not None:
        data = json.loads(Path(ride_path).read_text())
        if grounded is not None:
            data["grounded"] = grounded
        if same_source is not None:
            data["same_source"] = same_source
        Path(ride_path).write_text(json.dumps(data, indent=2) + "\n")
    card_path = next(export_dir.glob("*.card.txt"), None)
    stamp = None
    if card_path:
        stamp = card_mod.record_check_ride(card_path, flight.passed, flight.total)
    pair = flight.paired()
    set_state(run_dir, stage="done",
              flight={"passed": flight.passed, "total": flight.total,
                      # Where and how the numbers were measured. runtime was
                      # missing here, so every Teach Pigeon's report.json read
                      # "unrecorded" although the ride always ran in llama.cpp.
                      "runtime": flight.runtime,
                      "engine": flight.engine,
                      "constrained": flight.constrained,
                      "temperature": flight.temperature,
                      "check_rides": rides,
                      "graded_by_model": flight.graded_by_model,
                      "control_passed": flight.control_passed,
                      "significance": flight.significance_line(),
                      "power": flight.power_line(),
                      "paired": ({"fixed": pair[0], "broken": pair[1], "p": pair[2]}
                                 if pair else None),
                      "same_source": same_source,
                      "rows": [r.__dict__ for r in flight.rows],
                      "control_rows": [r.__dict__ for r in flight.control_rows]})
    # The control is emitted beside the score, never after it and never
    # optionally — the UI has no way to show one without the other.
    emit("flown", passed=flight.passed, total=flight.total,
         control_passed=flight.control_passed,
         significance=flight.significance_line(),
         power=flight.power_line(),
         graded_by_model=flight.graded_by_model, card=stamp)
    return flight


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    cfg = config_mod.load(Path(args.config))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        plan_report = planspec.validate(planspec.load_plan(run_dir / "plan.json"))
        if not plan_report.valid:
            emit("error", stage="plan",
                 msg="the plan fails validation: " +
                     "; ".join(f"{r.name}: {r.detail}" for r in plan_report.rejections))
            set_state(run_dir, error="plan invalid")
            return 2
        plan = plan_report.plan
        planspec.save_plan(plan, run_dir / "plan.json")   # persist the repairs
        set_state(run_dir, plan=plan)

        clock = Clock(run_dir)
        if stage_table(cfg, args, run_dir, plan, clock) is None:
            return 2
        return 0 if stage_fly(cfg, run_dir, plan, clock) else 2
    except Exception as exc:  # noqa: BLE001 — the UI needs the reason, not a stack-only exit
        import traceback
        traceback.print_exc()
        emit("error", stage="run", msg=str(exc)[:400])
        set_state(run_dir, error=str(exc)[:400])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
