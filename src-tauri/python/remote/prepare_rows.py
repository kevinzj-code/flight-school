#!/usr/bin/env python3
"""On the Mac, before a remote run: tokenize a training folder exactly as
`mlx_lm lora --mask-prompt` does, and write the plan the pod trains from.

Runs in the Mac's training venv (mlx-lm 0.31.3) and uses mlx-lm's own
tokenizer loader and ChatDataset, so the tokens are the ones a Mac run of the
same folder sees, by construction. It only tokenizes: mlx is pinned to the
CPU and nothing touches the GPU.

    prepare_rows.py <folder> <out_dir> [--iters N] [--model REPO]

Writes <out_dir>/tokens/{train,valid}.jsonl ({"t": [ids], "o": offset} per
row, in file order) and <out_dir>/plan.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote import rows as R  # noqa: E402

MAC_REPO = "mlx-community/Qwen3-VL-2B-Instruct-bf16"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--iters", type=int, help="default: two thirds of two passes over train.jsonl")
    ap.add_argument("--model", default=MAC_REPO, help="the repo whose tokenizer the Mac run uses")
    a = ap.parse_args(argv)

    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.utils import load_tokenizer
    import mlx_lm

    folder = a.folder.expanduser().resolve()
    counts = R.check_folder(folder)
    tokenizer = load_tokenizer(a.model)

    out = a.out_dir
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    lengths: dict[str, list[int]] = {}
    for split in ("train", "valid"):
        data = R.read_jsonl(folder / f"{split}.jsonl")
        ds = ChatDataset(data, tokenizer, mask_prompt=True)
        lines, lens = [], []
        for i, d in enumerate(data):
            tokens, offset = ds.process(d)
            mine = R.tokenize_row(tokenizer, d["messages"])
            if (list(tokens), offset) != mine:
                raise SystemExit(f"{split} row {i + 1}: rows.tokenize_row disagrees with mlx-lm's ChatDataset")
            if offset >= len(tokens):
                raise SystemExit(f"{split} row {i + 1}: nothing after the prompt to train on")
            lines.append(json.dumps({"t": list(tokens), "o": offset}, separators=(",", ":")))
            lens.append(len(tokens))
        (out / "tokens" / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
        lengths[split] = lens

    iters = a.iters or R.iters_for(counts["train"])
    plan = {
        "folder": str(folder),
        "rows": counts,
        "files": {f"{s}.jsonl": R.sha256(folder / f"{s}.jsonl") for s in R.SPLITS},
        "tokens": {f"{s}.jsonl": R.sha256(out / "tokens" / f"{s}.jsonl") for s in ("train", "valid")},
        "tokenizer": a.model,
        "mlx_lm": mlx_lm.__version__,
        "longest_row": max(lengths["train"] + lengths["valid"]),
        "max_seq_length": R.seq_limit(lengths["train"] + lengths["valid"]),
        "iters": iters,
        "batch_size": 1, "grad_accumulation_steps": 4, "learning_rate": 2e-5,
        "rank": 16, "scale": 20.0, "dropout": 0.0, "seed": 1,
        "steps_per_eval": 1000, "val_batches": 25, "save_every": 500, "steps_per_report": 10,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({k: plan[k] for k in ("rows", "longest_row", "max_seq_length", "iters")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
