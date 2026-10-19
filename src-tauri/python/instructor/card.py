"""The card that ships with every cartridge.

Plain text on purpose. It is the thing a person reads six months later to find
out what this file is, and the place where the awkward sentences have to appear
rather than being left out.

Two of those sentences are mandatory:

  - "Instructor was trained with help from a cloud model, once." The bootstrap
    dataset was drafted by a frontier model and edited by hand. The brand does
    not get to skip this line.
  - The STATE line. A cartridge that nobody has flown is a DRAFT, and says so.
    The spec is explicit: one button gets you a draft; the check ride is where
    it becomes real.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from . import __version__

HONESTY_LEDGER = "Instructor was trained with help from a cloud model, once"


def sha256(path: Path, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    read = 0
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if read >= limit:
                break
    return h.hexdigest()


@dataclass
class Card:
    job_name: str
    does: str
    kind: str
    modality: str
    base_path: Path
    base_sha: str
    adapter_path: Path
    train_examples: int
    holdout: int
    preset: str
    iters: int
    needs: str | None = None
    check_ride: str | None = None
    knowledge_file: str | None = None
    notes: list[str] = field(default_factory=list)
    measured_answer_tokens: int | None = None
    claimed_answer_tokens: int | None = None

    @property
    def state(self) -> str:
        return "FLOWN" if self.check_ride else "DRAFT — no check ride recorded"

    def render(self) -> str:
        w = 16
        def row(label: str, value: str) -> str:
            return f"{label:<{w}}{value}"

        lines = [
            row("CARTRIDGE:", self.job_name),
            row("STATE:", self.state),
            row("DOES:", self.does),
            row("KIND:", f"{self.kind} · {self.modality}"),
            row("BASE:", f"{self.base_path.name}  (sha256: {self.base_sha[:16]}…)"),
        ]
        if self.needs:
            lines.append(row("NEEDS:", self.needs))
        if self.knowledge_file:
            lines.append(row("READS:", f"{self.knowledge_file}  "
                                       f"(facts live here, not in the weights)"))
        lines.append(row("TRAINED ON:", f"{self.train_examples} examples, "
                                        f"{self.holdout} held out, "
                                        f"preset {self.preset} ({self.iters} iters)"))
        if self.measured_answer_tokens is not None:
            claim = self.claimed_answer_tokens
            note = f"{self.measured_answer_tokens} tokens measured"
            if claim is not None and claim != self.measured_answer_tokens:
                note += f"  (the plan claimed {claim}; measurement won)"
            lines.append(row("ANSWER LENGTH:", note))
        lines.append(row("CHECK RIDE:", self.check_ride or
                         "not flown — run `instructor fly` before trusting this"))
        lines.append(row("MADE WITH:", f"Flight School Instructor v{__version__}"))
        lines.append(row("", f"({HONESTY_LEDGER})"))
        lines.append(row("RUNS ON:", "plain llama.cpp — no app required"))
        if self.notes:
            lines.append("")
            lines.append("NOTES:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def from_plan(
    plan: dict[str, Any],
    *,
    base_path: Path,
    adapter_path: Path,
    train_examples: int,
    holdout: int,
    iters: int,
    mmproj: Path | None = None,
    notes: list[str] | None = None,
    measured_answer_tokens: int | None = None,
    claimed_answer_tokens: int | None = None,
) -> Card:
    return Card(
        job_name=plan["job_name"],
        does=plan["one_line"],
        kind=plan["cartridge_kind"],
        modality=plan.get("modality", "text"),
        base_path=Path(base_path),
        base_sha=sha256(base_path),
        adapter_path=Path(adapter_path),
        train_examples=train_examples,
        holdout=holdout,
        preset=plan["preset"],
        iters=iters,
        needs=f"{mmproj.name}  (vision)" if mmproj else None,
        knowledge_file=plan.get("knowledge_file"),
        notes=list(notes or []),
        measured_answer_tokens=measured_answer_tokens,
        claimed_answer_tokens=claimed_answer_tokens,
    )


def write(card: Card, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.render())
    return path


def record_check_ride(card_path: Path, passed: int, total: int,
                      overrides: int = 0, machine_passed: int | None = None) -> str:
    """Rewrite the CHECK RIDE and STATE lines after a flight.

    Only these two lines change. The card is not regenerated, because the rest
    of it describes a build that already happened.

    `overrides` is the number of verdicts a person changed by hand. It is
    written onto the card ON PURPOSE: a score a human adjusted is a different
    claim from one a grader produced alone, and the card is where that
    difference has to survive. The machine's own count is kept beside it.
    """
    path = Path(card_path)
    text = path.read_text()
    stamp = f"{passed} / {total} passed  ({date.today().isoformat()})"
    if overrides:
        stamp += (f"  [{overrides} verdict{'s' if overrides != 1 else ''} changed "
                  f"by hand; the grader scored "
                  f"{machine_passed if machine_passed is not None else '?'}]")
    out = []
    for line in text.splitlines():
        if line.startswith("CHECK RIDE:"):
            out.append(f"{'CHECK RIDE:':<16}{stamp}")
        elif line.startswith("STATE:"):
            out.append(f"{'STATE:':<16}FLOWN")
        else:
            out.append(line)
    path.write_text("\n".join(out) + "\n")
    return stamp
