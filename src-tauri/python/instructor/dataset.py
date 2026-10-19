"""Building the training files.

Four rules, each of which exists because breaking it caused a real failure:

1. THE HOLDOUT IS SPLIT FIRST. Before any file the trainer can see is written.
   A check ride against data the model trained on measures nothing.
2. THE SYSTEM PROMPT IS STRIPPED. The style rules were paid once during
   distillation; the answers now carry the behaviour on their own. This is the
   trick Flight School is built on — the brief never ships.
3. NOTHING IS TRUNCATED SILENTLY. Over-budget rows are dropped and named. A
   row cut off mid-answer teaches the model to stop mid-answer.
4. THE MODEL'S TOKEN CLAIM IS OVERRIDDEN BY MEASUREMENT, and both numbers are
   reported.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .generate import Row


@dataclass
class BuildResult:
    data_dir: Path
    train_path: Path
    valid_path: Path
    holdout_path: Path
    n_train: int
    n_valid: int
    n_holdout: int
    dropped: list[tuple[str, int]] = field(default_factory=list)
    duplicates: int = 0
    claimed_answer_tokens: int = 0
    measured_answer_tokens: int = 0
    max_seq_length: int = 0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _identity(question: str, image: str | None) -> tuple[str, str]:
    """What makes two rows "the same" row.

    Question text alone is wrong for vision: one phrasing ("What connector is
    this?") is deliberately reused across every image in the corpus, so a
    question-only key would collapse the whole dataset and make the leak check
    fire on every build. The image is the other half of the identity.
    """
    return (_norm(question), image or "")


#: The group the person's own examples land in. They are the standard the
#: rest of the set is drafted to match, and they never go into the holdout:
#: the test must be questions the person did not write the answers to.
YOUR_EXAMPLES = "your examples"


def split_holdout(rows: Sequence[Row], holdout_count: int,
                  seed: int = 11) -> tuple[list[Row], list[Row]]:
    """Stratified by group, so the check ride covers what the plan covered.

    A holdout drawn from one group would test one thing and be reported as if
    it tested the cartridge.
    """
    rng = random.Random(seed)
    by_group: dict[str, list[Row]] = {}
    for r in rows:
        by_group.setdefault(r.group, []).append(r)
    for group in by_group.values():
        rng.shuffle(group)

    holdout: list[Row] = []
    # The person's own rows are never drawn for the holdout — but they are
    # still training rows. The first version of this line dropped them from
    # BOTH, and the build's own note still said "in the set as written". The
    # test below exists because of that.
    names = sorted(g for g in by_group if g != YOUR_EXAMPLES)
    all_names = sorted(by_group)
    # Round-robin across groups until the quota is met, so every group that has
    # rows to spare contributes before any group contributes twice.
    #
    # The loop terminates on lack of PROGRESS, not on the pools being empty: a
    # group is never emptied into the holdout, so with one row per group every
    # pool stays non-empty forever and an emptiness test would spin. Asking for
    # a bigger holdout than the data can spare is a real thing a plan can do,
    # and it must return a short holdout rather than hang.
    while len(holdout) < holdout_count:
        took = 0
        for n in names:
            if len(holdout) >= holdout_count:
                break
            if len(by_group[n]) > 1:      # never empty a group into the holdout
                holdout.append(by_group[n].pop())
                took += 1
        if took == 0:
            break
    train = [r for n in all_names for r in by_group[n]]
    rng.shuffle(train)
    return train, holdout


def _messages(row: Row, *, for_vision: bool) -> dict:
    """One training example. NO SYSTEM TURN — see rule 2."""
    if for_vision and row.image:
        return {
            "images": [row.image],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": row.image},
                    {"type": "text", "text": row.question},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": row.answer},
                ]},
            ],
        }
    return {"messages": [
        {"role": "user", "content": row.question},
        {"role": "assistant", "content": row.answer},
    ]}


def write_dataset(
    rows: Sequence[Row],
    out_dir: Path,
    *,
    holdout_count: int,
    answer_token_lengths: Sequence[int],
    claimed_answer_tokens: int,
    max_seq_length: int,
    seed: int = 11,
    valid_fraction: float = 0.1,
    vision: bool = False,
    holdout: Sequence[dict] | None = None,
    strict_pin: bool = True,
) -> BuildResult:
    """Write train/valid/holdout. Returns exactly what was written and dropped.

    `holdout` PINS the held-out set: the rows with these identities are the
    holdout, verbatim, and nothing is re-split. A retrain that changes the
    test is two scores on two tests; the loop compares rounds over the same
    questions, so round 2 must be asked exactly what round 1 was.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(answer_token_lengths) != len(rows):
        raise ValueError("measurement and rows are out of step — refusing to write")

    # Defensive dedupe. generate.py already dedupes questions, but a duplicate
    # that survived to here would end up on both sides of the split and quietly
    # turn the check ride into a memorisation test.
    seen: set[tuple[str, str]] = set()
    unique_rows, unique_lengths, duplicates = [], [], 0
    for row, n in zip(rows, answer_token_lengths):
        key = _identity(row.question, row.image)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique_rows.append(row)
        unique_lengths.append(n)
    rows, answer_token_lengths = unique_rows, unique_lengths

    # Rule 4: measurement beats the claim, and the ceiling that actually gets
    # enforced is the generous one, so a good long answer is not thrown away
    # just because the model under-guessed its own verbosity.
    measured = max(answer_token_lengths) if answer_token_lengths else 0
    ceiling = max(claimed_answer_tokens, measured)

    # Rule 3: drop, and name. A row is over budget only if it exceeds the
    # ceiling AND is a genuine outlier against the rest of the set.
    keep: list[Row] = []
    dropped: list[tuple[str, int]] = []
    if answer_token_lengths:
        ordered = sorted(answer_token_lengths)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        outlier_cut = max(ceiling, int(p95 * 2))
    else:
        outlier_cut = ceiling
    for row, n in zip(rows, answer_token_lengths):
        if n > outlier_cut:
            dropped.append((row.question[:70], n))
        else:
            keep.append(row)

    # Rule 1: the split happens here, before a single trainable line is written.
    if holdout is not None:
        pinned = {_identity(h["question"], h.get("image")): i for i, h in enumerate(holdout)}
        found = {_identity(r.question, r.image): r for r in keep if _identity(r.question, r.image) in pinned}
        missing = [h["question"] for h in holdout if _identity(h["question"], h.get("image")) not in found]
        if missing and strict_pin:
            raise ValueError(f"{len(missing)} pinned held-out row(s) are not in the rows — "
                             f"refusing to retrain against a different test (first: {missing[0][:60]!r})")
        holdout_rows = [found[k] for k in sorted(found, key=pinned.__getitem__)]
        train_all = [r for r in keep if _identity(r.question, r.image) not in pinned]
        random.Random(seed).shuffle(train_all)
        holdout = holdout_rows
    else:
        train_all, holdout = split_holdout(keep, holdout_count, seed=seed)

    n_valid = max(1, int(len(train_all) * valid_fraction)) if len(train_all) > 1 else 0
    valid, train = train_all[:n_valid], train_all[n_valid:]

    # train/valid go in their OWN directory, and the holdout stays outside it.
    # mlx-lm looks for train.jsonl/valid.jsonl by name, but mlx-vlm hands the
    # path to `datasets.load_dataset`, which globs EVERY data file in the
    # folder. Sharing a directory with holdout.jsonl would quietly feed the
    # check ride into vision training.
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    holdout_path = out_dir / "holdout.jsonl"

    for path, subset in ((train_path, train), (valid_path, valid)):
        with path.open("w") as fh:
            for r in subset:
                fh.write(json.dumps(_messages(r, for_vision=vision)) + "\n")

    # The holdout keeps its ground truth and its group, because a person reads
    # it. It is never written in the trainer's format, so it cannot be fed in
    # by accident.
    with holdout_path.open("w") as fh:
        for r in holdout:
            fh.write(json.dumps(r.to_json()) + "\n")

    assert_no_leak(train_path, valid_path, holdout_path)

    return BuildResult(
        data_dir=data_dir,
        train_path=train_path, valid_path=valid_path, holdout_path=holdout_path,
        n_train=len(train), n_valid=len(valid), n_holdout=len(holdout),
        duplicates=duplicates,
        dropped=dropped,
        claimed_answer_tokens=claimed_answer_tokens,
        measured_answer_tokens=measured,
        max_seq_length=max_seq_length,
    )


def assert_no_leak(train: Path, valid: Path, holdout: Path) -> None:
    """The check that makes the check ride mean something.

    Cheap, and run on every build. If it ever fires, the run is wrong and no
    number it produced can be reported.
    """
    held = set()
    with holdout.open() as fh:
        for line in fh:
            obj = json.loads(line)
            held.add(_identity(obj["question"], obj.get("image")))
    for path in (train, valid):
        with path.open() as fh:
            for i, line in enumerate(fh, 1):
                obj = json.loads(line)
                image = (obj.get("images") or [None])[0]
                user = obj["messages"][0]["content"]
                if isinstance(user, list):
                    user = " ".join(c.get("text", "") for c in user
                                    if c.get("type") == "text")
                if _identity(user, image) in held:
                    raise RuntimeError(
                        f"holdout leak: {path.name} line {i} repeats a held-out "
                        f"question. The check ride would be measuring memorisation."
                    )


def read_holdout(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
