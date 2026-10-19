#!/usr/bin/env python3
"""On the pod: the Mac recipe (docs/TRAINING-RECIPE-QWEN3VL.md) in PyTorch + PEFT.

    train_remote.py <run_dir> [--model Qwen/Qwen3-VL-2B-Instruct] [--revision SHA]

<run_dir> holds what prepare_rows.py wrote on the Mac (plan.json, tokens/)
and the training folder as data/. The run writes, beside them:

    adapters/0000500_adapters.safetensors …   every save_every iterations
    adapters/adapters.safetensors             the last iteration
    adapters/adapter_config.json              mlx-lm's own config shape
    train.log, metrics.jsonl, timings.json, DONE

The adapters are written in mlx-lm's layout and names, so the Mac converts and
packs them along the same path as a Mac-trained adapter. The README beside
this file maps every setting to the mlx-lm line it copies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote import rows as R  # noqa: E402

BASE_REPO = "Qwen/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
# Every Linear mlx-lm wraps in a Qwen3-VL language layer (--num-layers -1, no keys).
TARGET = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
LAYERS, PER_LAYER = 28, 7


def log(f, line: str) -> None:
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def mlx_key(peft_name: str) -> tuple[str, str]:
    """`base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight`
    -> (`language_model.model.layers.0.self_attn.q_proj.lora_a`, "A"): the names
    the Mac's adapters carry (checked against patch-asks' 392 tensors)."""
    stem, rest = peft_name.split(".lora_", 1)
    which = rest[0]
    layer = stem.split("model.language_model.", 1)[1]
    return f"language_model.model.{layer}.lora_{which.lower()}", which


def mlx_state(model) -> dict:
    """PEFT's lora_A (r, in) and lora_B (out, r) as mlx-lm's lora_a (in, r) and
    lora_b (r, out): plain transposes (mlx_adapter_to_peft.py does the inverse)."""
    out = {}
    for name, p in model.named_parameters():
        if ".lora_" in name:
            key, _ = mlx_key(name)
            out[key] = p.detach().float().t().contiguous().cpu()
    return out


def adapter_config(plan: dict) -> dict:
    """The fields of the adapter_config.json mlx-lm writes, with this run's values."""
    return {"adapter_path": "adapters", "batch_size": plan["batch_size"], "data": "data",
            "fine_tune_type": "lora", "grad_accumulation_steps": plan["grad_accumulation_steps"],
            "grad_checkpoint": True, "iters": plan["iters"], "learning_rate": plan["learning_rate"],
            "lora_parameters": {"rank": plan["rank"], "dropout": plan["dropout"], "scale": plan["scale"]},
            "lr_schedule": None, "mask_prompt": True, "max_seq_length": plan["max_seq_length"],
            "model": "mlx-community/Qwen3-VL-2B-Instruct-bf16", "num_layers": -1, "optimizer": "adam",
            "save_every": plan["save_every"], "seed": plan["seed"], "steps_per_eval": plan["steps_per_eval"],
            "steps_per_report": plan["steps_per_report"], "train": True, "val_batches": plan["val_batches"],
            "trained_where": "remote: train_remote.py, torch + peft"}


class MlxAdam:
    """mlx.optimizers.Adam as mlx-lm builds it: betas (0.9, 0.999), eps 1e-8,
    bias_correction False (MLX's default; torch.optim.Adam always corrects).
    m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; p -= lr m / (sqrt(v) + eps)."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        import torch
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self, grads) -> None:
        import torch
        with torch.no_grad():
            for p, g, m, v in zip(self.params, grads, self.m, self.v):
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                p.sub_(self.lr * m / (v.sqrt() + self.eps))


def row_loss(model, tokens, offset, max_seq_length, device):
    """default_loss at batch 1: mean cross-entropy over the masked targets, and their count."""
    import torch
    import torch.nn.functional as F
    inputs, targets, mask = R.batch_arrays(tokens, offset, max_seq_length)
    x = torch.tensor([inputs], device=device)
    y = torch.tensor(targets, device=device)
    w = torch.tensor(mask, device=device, dtype=torch.float32)
    logits = model(input_ids=x, use_cache=False).logits[0].float()
    ce = F.cross_entropy(logits, y, reduction="none") * w
    n = w.sum()
    return ce.sum() / n, int(n.item())


def check_tokens(plan: dict, run: Path, tokenizer) -> int:
    """The pod's tokenizer against the Mac's tokens, row by row. Returns the rows that differ."""
    bad = 0
    for split in ("train", "valid"):
        data = R.read_jsonl(run / "data" / f"{split}.jsonl")
        mac = R.read_jsonl(run / "tokens" / f"{split}.jsonl")
        for d, m in zip(data, mac):
            t, o = R.tokenize_row(tokenizer, d["messages"])
            bad += (t != m["t"] or o != m["o"])
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default=BASE_REPO)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--first-eval-only", action="store_true", help="score the untrained model and stop")
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    run = a.run_dir
    plan = json.loads((run / "plan.json").read_text())
    for name, sha in plan["files"].items():
        if R.sha256(run / "data" / name) != sha:
            raise SystemExit(f"data/{name} is not the file the Mac prepared")
    for name, sha in plan["tokens"].items():
        if R.sha256(run / "tokens" / name) != sha:
            raise SystemExit(f"tokens/{name} is not the file the Mac prepared")
    R.check_folder(run / "data")

    out = run / "adapters"
    out.mkdir(exist_ok=True)
    f = open(run / "train.log", "a")
    log(f, f"torch {torch.__version__}, cuda {torch.version.cuda}, {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    bad = check_tokens(plan, run, tokenizer)
    log(f, f"tokens: {bad} rows differ between the pod's tokenizer and the Mac's")
    if bad:
        raise SystemExit("the pod tokenizes differently from the Mac; refusing to train on different rows")

    train = R.read_jsonl(run / "tokens" / "train.jsonl")
    valid = R.read_jsonl(run / "tokens" / "valid.jsonl")
    iters, msl, accum = plan["iters"], plan["max_seq_length"], plan["grad_accumulation_steps"]
    order, evals = R.mlx_order([len(r["t"]) for r in train], [len(r["t"]) for r in valid], iters,
                               seed=plan["seed"], steps_per_eval=plan["steps_per_eval"],
                               val_batches=plan["val_batches"])

    torch.manual_seed(plan["seed"])
    np.random.seed(plan["seed"])
    device = torch.device("cuda")
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16)
    model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=plan["rank"], lora_alpha=plan["scale"] * plan["rank"], lora_dropout=plan["dropout"],
                      target_modules=TARGET, bias="none")
    model = get_peft_model(model, lcfg)
    lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(lora) != 2 * LAYERS * PER_LAYER or any(".lora_" not in n or p.dtype != torch.float32 for n, p in lora):
        raise SystemExit(f"LoRA landed on {len(lora)} tensors, not {2 * LAYERS * PER_LAYER} fp32 lora_A/lora_B")
    n_train = sum(p.numel() for _, p in lora)
    log(f, f"Trainable parameters: {n_train / 1e6:.3f}M on {len(lora) // 2} modules, alpha {lcfg.lora_alpha} "
           f"(scale {plan['scale']} x rank {plan['rank']})")
    params = [p for _, p in lora]
    opt = MlxAdam(params, plan["learning_rate"])

    def evaluate(rows_idx):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in rows_idx:
                loss, k = row_loss(model, valid[i]["t"], valid[i]["o"], msl, device)
                tot += loss.item() * k
                n += k
        model.train()
        return tot / n

    def save(name):
        save_file(mlx_state(model), str(out / name))

    (out / "adapter_config.json").write_text(json.dumps(adapter_config(plan), indent=4))
    metrics = open(run / "metrics.jsonl", "a")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    grads = None
    losses, steps, trained, train_time = 0.0, 0, 0, 0.0
    t_start = time.time()
    for it in range(1, iters + 1):
        if it in evals:
            tic = time.perf_counter()
            vl = evaluate(evals[it])
            log(f, f"Iter {it}: Val loss {vl:.3f}, Val took {time.perf_counter() - tic:.3f}s")
            metrics.write(json.dumps({"it": it, "val_loss": vl}) + "\n")
            metrics.flush()
            if a.first_eval_only:
                return 0
        tic = time.perf_counter()
        r = train[order[it - 1]]
        loss, k = row_loss(model, r["t"], r["o"], msl, device)
        loss.backward()
        if it % accum == 0:
            grads = [p.grad / accum for p in params]
            opt.step(grads)
            for p in params:
                p.grad = None
        torch.cuda.synchronize()
        train_time += time.perf_counter() - tic
        losses += loss.item()
        steps += 1
        trained += k
        if it % plan["steps_per_report"] == 0 or it == iters:
            peak = torch.cuda.max_memory_allocated() / 1e9
            log(f, f"Iter {it}: Train loss {losses / steps:.3f}, Learning Rate {plan['learning_rate']:.3e}, "
                   f"It/sec {steps / train_time:.3f}, Trained Tokens {trained}, Peak mem {peak:.3f} GB")
            metrics.write(json.dumps({"it": it, "train_loss": losses / steps, "peak_gb": peak}) + "\n")
            metrics.flush()
            losses, steps, train_time = 0.0, 0, 0.0
        if it % plan["save_every"] == 0:
            save(f"{it:07d}_adapters.safetensors")
    # mlx-lm saves the final weights after the loop; an update never due (it % accum) stays unapplied.
    save("adapters.safetensors")
    wall = time.time() - t_start
    timings = {"iters": iters, "rows_per_second": round(iters / wall, 3), "wall_seconds": round(wall, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "note": "rows a second over the whole loop, evaluations and saves included"}
    (run / "timings.json").write_text(json.dumps(timings, indent=1) + "\n")
    log(f, f"Saved final weights to {out / 'adapters.safetensors'}. {json.dumps(timings)}")
    (run / "DONE").write_text(json.dumps({"plan_sha256": R.sha256(run / "plan.json")}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
