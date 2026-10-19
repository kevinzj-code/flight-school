"""What an mlx-lm 0.31.3 LoRA run sees, rebuilt without mlx: the tokens of each
row, where its loss starts, the order the rows come in, and which 25
validation rows each evaluation scores.

Shared by the Mac (prepare_rows.py, which tokenizes) and the pod
(train_remote.py, which trains). Nothing here imports mlx or torch, so the
same code decides both sides. Every rule names the mlx-lm line it copies; the
tests in bench/test_remote.py check the order against mlx-lm's own
iterate_batches.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "grade")
PAD_ID = 0          # trainer.py iterate_batches: batch_arr = np.zeros(...)
PAD_TO = 32         # trainer.py: pad_to = 32
SEQ_ROUND = 128     # the recipe: the longest templated row, rounded up to a multiple of 128


def tokenize_row(tokenizer, messages: list[dict]) -> tuple[list[int], int]:
    """datasets.py ChatDataset.process with mask_prompt: the whole chat, and the
    length of everything before the last turn with the generation prompt."""
    tokens = tokenizer.apply_chat_template(messages, return_dict=False)
    add_generation_prompt = messages[-1].get("role") == "assistant"
    offset = len(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=add_generation_prompt,
                                               return_dict=False))
    return list(tokens), offset


def seq_limit(lengths: list[int]) -> int:
    """`--max-seq-length`: the longest row, rounded up to a multiple of 128."""
    return SEQ_ROUND * math.ceil(max(lengths) / SEQ_ROUND)


def iters_for(train_rows: int) -> int:
    """`--iters` in rows (batch 1): two thirds of two passes since planner
    round 3, rounded to the nearest row, as the Mac runs were set: 6,227 rows
    -> 8,303 (8,302.67), 1,249 -> 1,665 (1,665.33), 1,299 -> 1,732."""
    return (8 * train_rows + 3) // 6


def batch_arrays(tokens: list[int], offset: int, max_seq_length: int) -> tuple[list[int], list[int], list[bool]]:
    """One row as trainer.py builds it at batch 1: inputs, targets, loss mask.

    The array is padded with id 0 to one plus the next multiple of 32 (capped at
    the sequence limit). default_loss scores targets at steps s = 1..T with
    offset <= s <= length, and target s is token s, so the step s = length is
    the first pad: mlx-lm trains every row to predict id 0 after its final
    token. The Mac log shows it (4 trained tokens per asks row: the label,
    <|im_end|>, the newline, the pad), so this copies it."""
    length = len(tokens)
    width = min(1 + PAD_TO * ((length + PAD_TO - 1) // PAD_TO), max_seq_length)
    kept = min(length, max_seq_length)
    arr = list(tokens[:kept]) + [PAD_ID] * (width - kept)
    inputs, targets = arr[:-1], arr[1:]
    mask = [offset <= s <= kept for s in range(1, len(targets) + 1)]
    return inputs, targets, mask


def mlx_order(train_lengths: list[int], valid_lengths: list[int], iters: int, *, seed: int = 1,
              steps_per_eval: int = 1000, val_batches: int = 25) -> tuple[list[int], dict[int, list[int]]]:
    """The row behind each iteration, and the validation rows of each
    evaluation, at batch 1 with no distributed group.

    lora.py seeds numpy's global generator once (np.random.seed(args.seed));
    trainer.py then draws a permutation each time an iterate_batches generator
    starts a pass. train() zips range(1, iters+1) with the endless training
    generator, so its first pass is drawn at it = 1 before the first
    evaluation, and each later pass is drawn when the previous one runs out,
    before that iteration's evaluation. Each evaluation (it == 1, every
    steps_per_eval, and the last) starts a fresh generator over the
    validation set and takes its first val_batches rows. Rows are sorted by
    token length first (sorted() is stable)."""
    rng = np.random.RandomState(seed)  # the legacy global generator np.random.seed reseeds
    train_sorted = sorted(range(len(train_lengths)), key=lambda i: train_lengths[i])
    valid_sorted = sorted(range(len(valid_lengths)), key=lambda i: valid_lengths[i])
    order: list[int] = []
    evals: dict[int, list[int]] = {}
    queue: list[int] = []
    for it in range(1, iters + 1):
        if not queue:
            queue = [train_sorted[i] for i in rng.permutation(len(train_sorted))]
            queue.reverse()
        order.append(queue.pop())
        if it == 1 or it % steps_per_eval == 0 or it == iters:
            perm = rng.permutation(len(valid_sorted))
            evals[it] = [valid_sorted[i] for i in perm[:val_batches]]
    return order, evals


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counted_rows(counts: dict, split: str) -> int | None:
    """counts.json writes a split either as a number or as {"all": n, ...}."""
    v = (counts.get("rows") or {}).get(split)
    if isinstance(v, dict):
        v = v.get("all")
    return v if isinstance(v, int) else None


def check_folder(folder: Path) -> dict[str, int]:
    """Refuse a partial folder: every split present, every line a chat, and the
    line counts equal to what counts.json says was written."""
    folder = Path(folder)
    counts_path = folder / "counts.json"
    if not counts_path.is_file():
        raise ValueError(f"{folder} has no counts.json, so a partial copy could not be told from a whole one")
    counts = json.loads(counts_path.read_text())
    got: dict[str, int] = {}
    for split in SPLITS:
        p = folder / f"{split}.jsonl"
        if not p.is_file():
            raise ValueError(f"{folder} has no {split}.jsonl")
        rows = read_jsonl(p)
        for i, r in enumerate(rows, 1):
            msgs = r.get("messages")
            if not isinstance(msgs, list) or not msgs or msgs[-1].get("role") != "assistant":
                raise ValueError(f"{split}.jsonl row {i} is not a chat ending in the answer")
        want = counted_rows(counts, split)
        if want is None:
            raise ValueError(f"counts.json has no row count for {split}")
        if len(rows) != want:
            raise ValueError(f"{split}.jsonl has {len(rows)} rows; counts.json says {want}")
        got[split] = len(rows)
    return got
