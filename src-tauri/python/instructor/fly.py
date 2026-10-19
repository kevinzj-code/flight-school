"""The check ride.

Deliberately dumb and visible. It loads the base plus the new adapter in plain
llama.cpp, asks the held-out questions, and prints every question, every answer
and every verdict. It prints a pass count. It does not print "done".

Grading is mechanical wherever the plan makes that possible — a fixed label set
means an exact match is the honest test. Where the answer is prose, a model
judges it against the rubric, and that verdict is LABELLED as a machine opinion
so nobody mistakes it for the human check the spec insists on.

No system prompt is sent. The training data had none stripped into it, so the
check ride must ask the way the training taught, or it measures the prompt
instead of the adapter.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import grade as grade_mod
from .config import Config
from .engine import LlamaServer, ServerSpec, serve, slots_for

# The measurement subsystem, ported from the earlier harness. `bench` is a sibling
# package of `instructor` under src-tauri/python, so it imports plainly once
# that directory is on the path — which it is, because quickbuild.py puts it
# there before importing either.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import mcnemar, mcnemar_floor          # noqa: E402
from bench.significance import verdict as _verdict  # noqa: E402

_LABEL_LIST = re.compile(r"(?:one of|exactly one of)\s*:?\s*(.+)", re.I)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def label_set(plan: dict) -> list[str] | None:
    """If output_shape names a fixed label set, return it. Else None."""
    m = _LABEL_LIST.search(plan.get("output_shape", ""))
    if not m:
        return None
    parts = [p.strip(" .").strip() for p in re.split(r",| or ", m.group(1))]
    labels = [p for p in parts if p and len(p) < 40]
    return labels if len(labels) >= 2 else None


def ride_user_texts(plan: dict, holdout: Sequence[dict[str, Any]], chunks: list[dict] | None = None) -> list[str]:
    """The text of each held-out row's user turn, built as spec §8.1–§8.3 has
    every Runner build it: the Pigeon's own template, the input trimmed and
    filled in one pass, a Reader's section block above it. (A picture travels
    as its own part, first.) Shared by the llama.cpp ride and the MLX ride."""
    from pigeon.manifest import prompt_template_for
    from pigeon.runner import fill, material_block, split_grounded
    grounded = bool(plan.get("grounded"))
    template = prompt_template_for(any(h.get("image") for h in holdout), grounded)
    out = []
    for item in holdout:
        question = item["question"]
        found = split_grounded(question, chunks or []) if grounded else None
        if not grounded:
            out.append(fill(template, question))
        elif found is None:
            # Not one of the material's sections (or no material given): the
            # row is sent as written, trimmed.
            out.append(question.strip())
        else:
            out.append(fill(template, found[1], material_block(found[0])))
    return out


def engine_rows(plan: dict, holdout: Sequence[dict[str, Any]], chunks: list[dict] | None = None) -> list[dict]:
    """The held-out rows as the engine is asked them (`engine_ride.holdout_row`'s
    table shape): the bare question, which the engine fills into the file's
    own template; a Reader's section given by ordinal (E25), split back out of
    the row as `ride_user_texts` does."""
    from pigeon.runner import split_grounded
    out = []
    for item in holdout:
        row = {"input": item["question"], "expected": item.get("answer")}
        if item.get("image"):
            row["image"] = item["image"]
        found = split_grounded(item["question"], chunks or []) if plan.get("grounded") else None
        if found is not None:
            row["input"], row["sections"] = found[1], [found[0]["ordinal"]]
        out.append(row)
    return out


def label_grammar(labels: Sequence[str]) -> str:
    """llama.cpp GBNF admitting exactly one line of the label set (spec §8.4).
    JSON string quoting is valid GBNF literal quoting for these characters."""
    return "root ::= " + " | ".join(json.dumps(str(l)) for l in labels)


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


@dataclass
class FlightRow:
    question: str
    expected: str
    actual: str
    group: str
    passed: bool
    how: str            # "rules" | "exact" | "judge" | "judge-error"
    image: str | None = None
    # A Reader's row: whether the answer leans on the expected answer's
    # sentence (pigeon/support.py `same_source`), or None for other kinds.
    same_source: dict | None = None


@dataclass
class Flight:
    rows: list[FlightRow] = field(default_factory=list)
    graded_by_model: int = 0
    #: The same questions answered with the adapter switched OFF. Empty only
    #: when the control was explicitly skipped — see `run_flight`.
    control_rows: list[FlightRow] = field(default_factory=list)
    #: WHICH RUNTIME PRODUCED THESE NUMBERS. Constant here because this module
    #: only ever asks llama.cpp — the same engine the sibling apps serve with,
    #: on the converted GGUF adapter, which is the whole point of flying after
    #: converting rather than before.
    #:
    #: It travels into report.json because a score measured in one runtime does
    #: not describe another. Measured: an adapter that reproduces its corpus in
    #: MLX returned 0/20 through llama.cpp on the same questions (the baseline notes,
    #: "The check ride was never the hole"). A scorecard that does not say where
    #: it was measured is a scorecard that can be wrong without looking wrong.
    runtime: str = "llama.cpp"
    #: D98: the pigeon-engine that answered (its version stamp), when the
    #: ride ran through the engine rather than this module's own server.
    engine: str | None = None
    #: Whether every answer (adapter and control alike) was limited to the
    #: label set while it was generated — spec §8.4. The manifest may only say
    #: `constrained` when the ride that produced its score decoded that way.
    constrained: bool = False
    #: The temperature every question was asked at (spec §5.5, §12.1).
    temperature: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.rows if r.passed)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def control_passed(self) -> int | None:
        """The score with training removed. `None` means it was not measured —
        which is a different thing from zero, and is displayed differently."""
        return sum(1 for r in self.control_rows if r.passed) if self.control_rows else None

    def paired(self) -> tuple[int, int, float] | None:
        """(fixed, broken, p) for adapter against control, or None without one.

        Raw totals lie at this sample size: run-to-run noise is about one case
        on a 47-case gate. What the score means is decided here, not by
        subtracting two numbers.
        """
        if not self.control_rows or len(self.control_rows) != len(self.rows):
            return None
        return mcnemar([r.passed for r in self.control_rows],
                       [r.passed for r in self.rows])

    def significance_line(self) -> str:
        """One line for the report, and it refuses to say "better" on a result
        that has not cleared the bar."""
        pair = self.paired()
        if pair is None:
            return ("no control was measured — this score cannot tell "
                    "\"the adapter learned the job\" from \"the prompt got shorter\"")
        fixed, broken, p = pair
        return _verdict(fixed, broken, p)

    def power_line(self) -> str:
        """What this ride COULD have proved, before it proved anything.

        A ride's discriminating power is bounded by how many cases the control
        gets wrong, not by how many questions it asks.
        """
        pair = self.paired()
        if pair is None:
            return "power: unknown without a control"
        n = pair[0] + pair[1]
        floor = mcnemar_floor(n)
        if floor >= 0.05:
            return (f"power: {n} case(s) differ — the best p reachable is "
                    f"{floor:.3g}, so this ride could not have shown significance "
                    f"however it landed. Six differing cases is the minimum")
        return f"power: {n} case(s) differ — enough to reach p = {floor:.3g} at best"

    def by_group(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.rows:
            slot = out.setdefault(r.group, [0, 0])
            slot[0] += int(r.passed)
            slot[1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}


JUDGE_SYSTEM = (
    "You grade one answer against a rubric. You reply with exactly one word: "
    "PASS or FAIL. No explanation, no punctuation."
)


def _judge(srv: LlamaServer, plan: dict, question: str,
           expected: str, actual: str, own_server: bool = True) -> bool:
    rubric = "\n".join(f"- {r}" for r in plan["grading_rubric"])
    user = (
        f"The job: {plan['one_line']}\n"
        f"Required output shape: {plan['output_shape']}\n\n"
        f"Rubric — every line must hold:\n{rubric}\n\n"
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Answer being graded: {actual}\n\n"
        f"Does the graded answer satisfy every rubric line? PASS or FAIL."
    )
    # The judge must not be the cartridge grading its own homework. The server
    # has the adapter resident, so this request switches it off for the judge
    # call only — per-request lora scale 0 gives the clean base (ADAPTERS.md
    # §4c). Same server, same weights, different behaviour per request.
    verdict, _ = srv.chat_full(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=6, temperature=0.0, seed=11, think=False,
        # Only meaningful when the judge shares a server with the cartridge:
        # switch the adapter off so it does not grade its own homework. A
        # separate judge server has no adapter loaded at all.
        lora=[{"id": 0, "scale": 0.0}] if own_server else None,
    )
    return verdict.strip().upper().startswith("PASS")


def run_flight(
    srv: LlamaServer,
    plan: dict,
    holdout: Sequence[dict[str, Any]],
    *,
    max_answer_tokens: int,
    judge: bool = True,
    judge_srv: LlamaServer | None = None,
    on_row: Callable[[FlightRow], None] | None = None,
    control: bool = True,
    on_control_row: Callable[[FlightRow], None] | None = None,
    timings: dict[str, float] | None = None,
    chunks: list[dict] | None = None,
    answers: Callable[[bool], list[str]] | None = None,
) -> Flight:
    """Ask the held-out questions and grade the answers.

    `judge_srv` exists because of a real limit: where the answer is prose there
    is no label to match, so a model grades it — and grading with the SAME model
    that produced the answer is close to meaningless when that model is a 0.5B.
    The judge is a separate, stronger server when one is supplied; the engine
    lock means the two run in sequence, not side by side.

    `control` runs the whole ride a second time with the adapter switched off,
    and it defaults ON because a score without it is not a result. The measured
    spread on one job was 46/47 with the adapter, 5/47 with the trained shape
    and no adapter, 34/47 with the full prompt and no adapter — without the
    middle number there is no way to tell which of those a score is.

    `answers(control)` gives every row's answer, in holdout order, from
    somewhere else: the engine (DEV-6), which asked the packed file and its
    D83 control. The grading below is the same either way; `srv` then serves
    only a judge, if one is needed.

    THE PASSES NEVER INTERLEAVE, and that is a correctness requirement, not
    tidiness: reusing KV across requests that differ in adapter state can carry
    the other configuration's state over (measured, ADAPTERS.md). The engine
    refuses caching for any request naming an adapter, and the control pass runs
    whole, after the adapter pass has finished.
    """
    labels = label_set(plan)
    rule = grade_mod.rule_for(plan)
    flight = Flight()
    # Spec D20/D27: a label answer is generated inside the label set, in the
    # ride exactly as a Runner will generate it. A rules-graded plan (a reasoned
    # answer, a recipe) is not constrained here: its grammar is not measured yet.
    # A plan can opt out with "constrained": false.
    grammar = label_grammar(labels) if labels and rule is None and plan.get("constrained", True) else None
    flight.constrained = grammar is not None
    # The file's output_contract.max_tokens: a Runner stops there (§8.5), so the
    # ride does too. It once allowed twice the answer, and recorded answers
    # longer than any Runner can give.
    budget = max_answer_tokens
    holdout = list(holdout)
    pending: list[tuple[int, FlightRow]] = []

    texts = ride_user_texts(plan, holdout, chunks)

    def _ask_with(i: int, lora: list[dict] | None) -> tuple[int, str]:
        item = holdout[i]
        image = item.get("image")
        if image:
            content: Any = [
                {"type": "image_url", "image_url": {"url": _data_uri(Path(image))}},
                {"type": "text", "text": texts[i]},
            ]
        else:
            content = texts[i]
        # No system prompt. The training had none; the check ride has none.
        #
        # `lora` is None for the adapter pass and [{"id": 0, "scale": 0.0}] for
        # the control — the same per-request switch the judge uses, so the
        # control is the SAME server and the SAME weights with the training
        # turned off, rather than a second model that would differ in other ways
        # too. The engine turns prompt caching off for any request that names an
        # adapter, which is what makes asking identical prompts under two
        # adapter states safe.
        extra = {"grammar": grammar} if grammar else {}
        return i, srv.chat(
            [{"role": "user", "content": content}],
            max_tokens=budget, temperature=0.0, seed=11, think=False,
            lora=lora, **extra,
        ).strip()

    def ask(i: int) -> tuple[int, str]:
        return _ask_with(i, None)

    def ask_control(i: int) -> tuple[int, str]:
        return _ask_with(i, [{"id": 0, "scale": 0.0}])

    def grade(i: int, row: FlightRow) -> tuple[int, bool | None]:
        """None means the judge itself failed — a different thing from FAIL."""
        try:
            return i, _judge(judge_srv or srv, plan, row.question, row.expected,
                             row.actual, own_server=judge_srv is None)
        except Exception:
            return i, None

    # Rows are pre-allocated and filled BY INDEX so that fanning the requests
    # out cannot reorder check-ride.json, while on_row still fires as each one
    # lands. Every request in a pass is identical in adapter state — the answer
    # pass uses the cartridge, the grading pass switches it off — so the two
    # never interleave, which is the one combination reused KV is unsafe for.
    slots = max(1, getattr(getattr(srv, "spec", None), "n_parallel", 1))
    if answers is not None:
        given = {False: answers(False), True: answers(True) if control else None}
        ask = lambda i: (i, given[False][i].strip())  # noqa: E731
        ask_control = lambda i: (i, given[True][i].strip())  # noqa: E731
    flight.rows = [None] * len(holdout)                      # type: ignore[list-item]
    # Wall time per pass, answers plus any grading on this server, for the
    # caller's timings.json. Measurement only — nothing here reads it back.
    clock = time.monotonic()

    def collect(i: int, actual: str) -> None:
        item = holdout[i]
        expected = item["answer"]
        passed, how = grade_mod.grade_row(expected, actual, labels=labels,
                                          rule=rule, judge=judge, norm=_norm)
        row = FlightRow(question=item["question"], expected=expected, actual=actual,
                        group=item.get("group", ""), passed=passed, how=how,
                        image=item.get("image"))
        flight.rows[i] = row
        if how == "judge":
            pending.append((i, row))
        elif on_row:
            on_row(row)

    _fan_out(ask, range(len(holdout)), slots, lambda r: collect(*r))

    # Second pass: everything that needs a model's opinion, graded together —
    # by `judge_srv` if one was supplied, otherwise by the cartridge's own
    # server with the adapter switched off.
    if pending:
        grader_slots = max(1, getattr((judge_srv or srv).spec, "n_parallel", 1))

        def apply(result: tuple[int, bool | None]) -> None:
            i, verdict = result
            row = flight.rows[i]
            if verdict is None:
                row.how, row.passed = "judge-error", False
            else:
                row.passed = verdict
                flight.graded_by_model += 1
            if on_row:
                on_row(row)

        _fan_out(lambda pair: grade(*pair), pending, grader_slots, apply)

    if timings is not None:
        timings["adapter"] = time.monotonic() - clock
        clock = time.monotonic()

    # -----------------------------------------------------------------------
    # THE CONTROL. The same questions, the same grading, the training switched
    # off. It runs whole and AFTER the adapter pass — never interleaved — for
    # the KV reason in this function's docstring.
    if control:
        control_rows: list[FlightRow | None] = [None] * len(holdout)
        control_pending: list[tuple[int, FlightRow]] = []

        def collect_control(i: int, actual: str) -> None:
            item = holdout[i]
            expected = item["answer"]
            # The SAME call as the adapter pass, not a copy of it. A control
            # graded more leniently would flatter the adapter, and one graded
            # more strictly would invent a win.
            passed, how = grade_mod.grade_row(expected, actual, labels=labels,
                                              rule=rule, judge=judge, norm=_norm)
            row = FlightRow(question=item["question"], expected=expected,
                            actual=actual, group=item.get("group", ""),
                            passed=passed, how=how, image=item.get("image"))
            control_rows[i] = row
            if how == "judge":
                control_pending.append((i, row))
            elif on_control_row:
                on_control_row(row)

        _fan_out(ask_control, range(len(holdout)), slots,
                 lambda r: collect_control(*r))

        if control_pending:
            grader_slots = max(1, getattr((judge_srv or srv).spec, "n_parallel", 1))

            def apply_control(result: tuple[int, bool | None]) -> None:
                i, v = result
                row = control_rows[i]
                if v is None:
                    row.how, row.passed = "judge-error", False
                else:
                    row.passed = v
                if on_control_row:
                    on_control_row(row)

            def grade_control(i: int, row: FlightRow) -> tuple[int, bool | None]:
                try:
                    return i, _judge(judge_srv or srv, plan, row.question,
                                     row.expected, row.actual,
                                     own_server=judge_srv is None)
                except Exception:
                    return i, None

            _fan_out(lambda pair: grade_control(*pair), control_pending,
                     grader_slots, apply_control)

        flight.control_rows = [r for r in control_rows if r is not None]
        if timings is not None:
            timings["control"] = time.monotonic() - clock

    return flight


def _fan_out(work, items, workers: int, on_result) -> None:
    """Run `work` over `items` at most `workers` at a time, handing each result
    to `on_result` as it lands.

    Sequential when workers == 1 — which is what a machine without memory
    headroom gets, and exactly the behaviour this code had before slots
    existed. No result is dropped and no exception is swallowed: a raising task
    propagates once the in-flight work has drained.

    `on_result` runs on the CALLING thread, never on a worker. That is what
    lets the callers mutate shared state — filling a row slot, appending to the
    pending list, counting graded rows — without a lock. Only `work` is
    concurrent, and `work` here is one HTTP request.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        for item in items:
            on_result(work(item))
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        for fut in as_completed([pool.submit(work, item) for item in items]):
            on_result(fut.result())


def answer_budget(adapter: Path, plan: dict) -> int:
    """How many tokens a check-ride answer may use: the Pigeon's own
    `output_contract.max_tokens` (`pigeon.manifest.answer_cap`), because a
    Runner stops there and the ride is a Runner.

    The plan's `max_answer_tokens` is a CLAIM the model made before any data
    existed; the build measured the truth. Sizing the ride from the claim
    starves a cartridge whose answers are longer than predicted — measured on
    the poet, where a 120-token claim against 303-token real answers cut off
    14 of 20 rides mid-sentence and failed them on a completeness rule.
    """
    import json as _json
    from pigeon.manifest import answer_cap
    measured = 0
    measured_file = Path(adapter).parent.parent / "measured.json"
    if measured_file.exists():
        try:
            measured = int(_json.loads(measured_file.read_text()).get("answer_tokens_measured") or 0)
        except (OSError, ValueError, TypeError):
            pass
    return answer_cap(plan, measured, grounded=bool(plan.get("grounded")))


def serving_base(cfg: Config, adapter: Path) -> Path:
    """The base this adapter was trained against, recorded at build time.

    Falls back to the pilot's target base for runs that predate base.json.
    An adapter only fits the base it was trained on; llama.cpp refuses the
    mismatch at best and is quietly wrong at worst, so this is read, not assumed.
    """
    import json as _json
    base_file = Path(adapter).parent.parent / "base.json"
    if base_file.exists():
        try:
            rec = _json.loads(base_file.read_text())
            p = Path(rec["student_gguf"])
            if p.exists():
                return p
        except (OSError, ValueError, KeyError):
            pass
    return cfg.target_gguf


def judge_server(cfg: Config, which: str = "target", ctx: int = 4096):
    """A clean, stronger model to grade prose answers. No adapter loaded."""
    model = {"target": cfg.target_gguf, "instructor": cfg.instructor_gguf,
             "demo": cfg.demo_gguf}.get(which) or cfg.target_gguf
    spec = ServerSpec(model=model, ctx=ctx, n_parallel=slots_for(cfg, model))
    return serve(cfg, spec, f"grading ({model.name})",
                 log=cfg.runs_dir / "judge-server.log")


def flight_server(cfg: Config, adapter: Path, *, vision: bool, ctx: int = 8192):
    base = serving_base(cfg, Path(adapter))
    spec = ServerSpec(
        model=base,
        ctx=ctx,
        lora=Path(adapter),
        mmproj=cfg.target_mmproj if vision else None,
        n_parallel=slots_for(cfg, base),
    )
    return serve(cfg, spec, "check ride", log=cfg.runs_dir / "fly-server.log")


def save_flight(flight: Flight, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pair = flight.paired()
    path.write_text(json.dumps({
        "passed": flight.passed, "total": flight.total,
        "graded_by_model": flight.graded_by_model,
        "by_group": flight.by_group(),
        # The control travels with the score or the score means nothing later.
        # `null` is deliberately distinct from 0: not measured is not the same
        # as measured at zero, and the UI shows the two differently.
        "control_passed": flight.control_passed,
        "runtime": flight.runtime,
        **({"engine": flight.engine} if flight.engine else {}),
        "constrained": flight.constrained,
        "temperature": flight.temperature,
        "significance": flight.significance_line(),
        "power": flight.power_line(),
        "paired": ({"fixed": pair[0], "broken": pair[1], "p": pair[2]}
                   if pair else None),
        "rows": [r.__dict__ for r in flight.rows],
        "control_rows": [r.__dict__ for r in flight.control_rows],
    }, indent=2) + "\n")
    return path
