"""Real tokenized lengths, and the memory verdict.

Ported from Flight School's `lesson.rs::lesson_preflight` and `memory_verdict`,
including the guard that file exists for. Quoting its comment, because the bug
is worth remembering:

    apply_chat_template(tokenize=True) does NOT return a plain list on every
    transformers version: 5.x hands back a BatchEncoding, whose len() is the
    number of KEYS (2 — input_ids, attention_mask). Taking that as a token
    count silently made every lesson's ceiling 128 and truncated every
    training example.

Every number here is measured through the base model's own tokenizer. Nothing
in this module estimates a token count from character length.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from typing import Sequence

from .config import Config

# Measured through the training venv's transformers. Two lengths per row: the
# whole rendered example (sets max_seq_length) and the assistant turn alone
# (overrides the plan's max_answer_tokens claim).
_MEASURE = r'''
import sys, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
rows = json.load(sys.stdin)

def n_tokens(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "input_ids"):                       # BatchEncoding (transformers >= 5)
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):  # batched shape
        ids = ids[0]
    return len(ids)

full, answer = [], []
for r in rows:
    full.append(n_tokens([{"role": "user", "content": r["q"]},
                          {"role": "assistant", "content": r["a"]}]))
    answer.append(len(tok.encode(r["a"], add_special_tokens=False)))

# Cross-check against a raw encode of the rendered text. If the two disagree
# wildly, the template path is returning something this script does not
# understand, and a wrong ceiling here truncates silently — so fail loudly.
probe = [{"role": "user", "content": rows[0]["q"]},
         {"role": "assistant", "content": rows[0]["a"]}]
raw = len(tok.encode(tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)))
if full[0] < raw / 2:
    print(json.dumps({"err": f"token count {full[0]} disagrees with raw encode {raw}"}))
else:
    print(json.dumps({"full": full, "answer": answer}))
'''


class PreflightError(RuntimeError):
    pass


def seq_ceiling(longest: int) -> int:
    """longest + 25 percent, rounded up to a multiple of 128."""
    padded = math.ceil(longest * 1.25)
    return max(1, math.ceil(padded / 128)) * 128


@dataclass
class Preflight:
    full_lengths: list[int]
    answer_lengths: list[int]
    max_seq_length: int
    est_memory_gb: float
    budget_gb: float
    verdict: str                     # fits | tight | wont_fit
    grad_checkpoint: bool
    longest_rows: list[str] = field(default_factory=list)

    @property
    def longest(self) -> int:
        return max(self.full_lengths) if self.full_lengths else 0

    def percentile(self, p: float) -> int:
        if not self.full_lengths:
            return 0
        s = sorted(self.full_lengths)
        return s[min(len(s) - 1, int(len(s) * p))]


def measure(cfg: Config, repo: str, rows: Sequence[dict[str, str]], *,
            extra_tokens: int = 0) -> Preflight:
    """rows are {"q": question, "a": answer}. Raises rather than guessing.

    `extra_tokens` is what every row carries that the tokenizer cannot see —
    a picture job's image allowance. It goes into the window BEFORE the memory
    verdict, because a verdict on the text alone is a verdict on a window the
    run never trains at."""
    if not rows:
        raise PreflightError("nothing to measure")
    if not cfg.venv_train.exists():
        raise PreflightError(f"training venv missing at {cfg.venv_train}")

    proc = subprocess.run(
        [str(cfg.venv_train), "-c", _MEASURE, repo],
        input=json.dumps(list(rows)),
        capture_output=True, text=True, env=cfg.train_env(), timeout=900,
    )
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out:
        tail = (proc.stderr or "").strip().splitlines()
        raise PreflightError(
            f"could not render the chat template for {repo}: "
            f"{tail[-1] if tail else 'no output'}"
        )
    try:
        data = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise PreflightError(f"tokenizer returned unparseable output: {out[:200]}") from exc
    if "err" in data:
        raise PreflightError(
            f"The tokenizer returned a length this build doesn't understand "
            f"({data['err']}). Training would silently truncate every example, "
            f"so it is refused here."
        )

    full, answer = data["full"], data["answer"]
    max_seq, est, budget, verdict, grad_ckpt = shape_for(
        model_dims(cfg, repo), max(full), extra_tokens)

    ranked = sorted(zip(full, rows), key=lambda t: -t[0])[:3]
    longest_rows = [f"({n} tok) {r['q'][:80]}" for n, r in ranked]

    return Preflight(full, answer, max_seq, est, budget, verdict, grad_ckpt, longest_rows)


def shape_for(dims: ModelDims | None, longest: int, extra_tokens: int = 0,
              budget_gb: float | None = None) -> tuple[int, float, float, str, bool]:
    """The window, the memory verdict and checkpointing for the longest row.

    Try the fast shape first: no checkpointing. A run pays the recompute cost
    only when the estimate without it crowds the budget — so a long window
    keeps checkpointing on a 16 GB Mac and drops it on a Mac with the room.
    That estimate is already 3x pessimistic and must sit under 80% of the
    budget to count as "fits". On a picture job the image tokens are in the
    window and pass through the language model like any other; the vision
    tower is frozen and runs forward only.

    A hybrid base (Qwen3.5's linear attention) keeps the older rule: off only
    at a window of 1024 or under. Phase A measured checkpointing making long
    windows on that base FASTER as well as smaller, so room to drop it is not a
    reason to (the training-efficiency notes, section 2).
    """
    max_seq = seq_ceiling(longest + extra_tokens)
    est_off, budget, verdict_off = memory_verdict(dims, max_seq, False, budget_gb)
    grad_ckpt = verdict_off != "fits" or (dims is not None and dims.is_hybrid and max_seq > 1024)
    est, budget, verdict = (memory_verdict(dims, max_seq, True, budget_gb) if grad_ckpt
                            else (est_off, budget, verdict_off))
    return max_seq, est, budget, verdict, grad_ckpt


# ---------------------------------------------------------------------------
# memory arithmetic — ported from lesson.rs::memory_verdict


@dataclass
class ModelDims:
    hidden_size: int
    num_layers: int
    weights_gb: float
    vocab_size: int = 0
    # Qwen3.5 and friends interleave GatedDeltaNet linear attention with full
    # attention. config.json advertises it as `layer_types` or
    # `full_attention_interval`; both mean backprop here is expensive.
    is_hybrid: bool = False


def model_dims(cfg: Config, repo: str) -> ModelDims | None:
    """Read hidden size, depth and on-disk weight size from the MLX snapshot."""
    slug = "models--" + repo.replace("/", "--")
    snaps = cfg.hf_home / "hub" / slug / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), reverse=True):
        cfg_file = snap / "config.json"
        if not cfg_file.exists():
            continue
        try:
            raw = json.loads(cfg_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Qwen3.5 nests the language model under text_config.
        block = raw.get("text_config", raw)
        hidden = block.get("hidden_size")
        layers = block.get("num_hidden_layers")
        if not hidden or not layers:
            continue
        weights = sum(f.stat().st_size for f in snap.glob("*.safetensors")) / 1e9
        vocab = block.get("vocab_size") or raw.get("vocab_size") or 0
        hybrid = bool(block.get("full_attention_interval")
                      or "linear_attention" in (block.get("layer_types") or []))
        return ModelDims(int(hidden), int(layers), weights, int(vocab), hybrid)
    return None


def fitting_ceiling(dims: ModelDims | None) -> int:
    """The largest sequence window (multiple of 128) this machine can train
    this base at, checkpointing on. 0 when even 128 doesn't fit — or when the
    shape is unknown, because a cap derived from a guess is worse than a
    refusal."""
    if dims is None:
        return 0
    best = 0
    for seq in range(128, 8193, 128):
        _, _, verdict = memory_verdict(dims, seq, True)
        if verdict == "wont_fit":
            break
        best = seq
    return best


def fit_to_budget(pf: "Preflight", dims: ModelDims | None,
                  rows: list, min_rows: int) -> tuple["Preflight", list, list, str]:
    """When the measured window doesn't fit, cap it at what DOES and drop the
    rows that no longer do — visibly. Returns (new preflight, kept rows,
    dropped [(question, tokens)], warning sentence). Raises when capping can't
    save enough data, with the reason in the message.

    This exists because the hard refusal was a dead end with no lever: a
    persona whose creative drafting produced a handful of long answers refused
    the WHOLE build over rows the training could simply do without. Dropping
    is rule 3's shape — named, never silent — applied one stage earlier.
    """
    cap = fitting_ceiling(dims)
    if cap < 256:
        raise PreflightError(
            f"this base cannot be trained on this machine at any useful window "
            f"(the largest that fits is {cap} tokens). "
            f"{pf.est_memory_gb:.1f} GB estimated against {pf.budget_gb:.1f} GB.")
    keep, kept_full, kept_answer, dropped = [], [], [], []
    for row, n_full, n_answer in zip(rows, pf.full_lengths, pf.answer_lengths):
        if n_full > cap:
            dropped.append((getattr(row, "question", str(row))[:70], n_full))
        else:
            keep.append(row)
            kept_full.append(n_full)
            kept_answer.append(n_answer)
    if len(keep) < min_rows:
        raise PreflightError(
            f"capping the window at {cap} tokens (the most this machine can "
            f"train) would keep only {len(keep)} of {len(rows)} rows — too few "
            f"to train on. Shorten the answers or lower max_answer_tokens in "
            f"the plan.")
    est, budget, verdict = memory_verdict(dims, cap, True)
    capped = Preflight(kept_full, kept_answer, cap, est, budget, verdict,
                       grad_checkpoint=True, longest_rows=pf.longest_rows)
    warn = (f"the full set needs {pf.est_memory_gb:.1f} GB this machine doesn't "
            f"have — capping the window at {cap} tokens and dropping "
            f"{len(dropped)} over-long row{'s' if len(dropped) != 1 else ''} "
            f"(named below) so the other {len(keep)} can train")
    return capped, keep, dropped, warn


def memory_verdict(dims: ModelDims | None, max_seq: int, grad_checkpoint: bool,
                   budget_gb: float | None = None) -> tuple[float, float, str]:
    """fits / tight / won't fit, as arithmetic rather than vibes.

    The checkpointing-ON coefficient was fitted to Flight School's measured runs
    (the Phase A results); OFF holds activations instead of recomputing and is
    taken at 3x — deliberately pessimistic, since an under-estimate here is the
    swap-thrash failure and an over-estimate merely leaves checkpointing on.

    `budget_gb` defaults to this machine's; a test passes one in.
    """
    if budget_gb is None:
        import os
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            total_bytes = 17_179_869_184
        budget_gb = (total_bytes / 1e9) * 0.75 - 1.5
    if dims is None:
        return 0.0, budget_gb, "unknown"
    bytes_per = 24.0 if grad_checkpoint else 72.0
    act_gb = max_seq * dims.hidden_size * dims.num_layers * bytes_per / 1e9

    # CORRECTION. An earlier version of this file added a "logits term" here on
    # the theory that a 248k vocabulary made the loss dominate. That was WRONG,
    # and Flight School's own the Phase A results settles it: Qwen2.5-1.5B,
    # 28 layers, seq 2560 measured 4.5 GB, and the formula WITHOUT any logits
    # term predicts 4.51 GB. Exact. Adding logits over-predicted by 6 GB.
    #
    # The real cause of the Instructor run's 11.6 GB and 20 s/iteration is the
    # one PHASE-A already measured and named: Qwen3.5 is a HYBRID model, and
    # backpropagation through its GatedDeltaNet linear-attention layers is
    # pathological in MLX. Their table, same machine:
    #
    #   Qwen3.5-2B, last 8 layers, seq 1280   50 s/iter   13.2 GB
    #   Qwen2.5-1.5B, all 28 layers, seq 2560  ~4 s/iter    4.5 GB
    #
    # 60x slower and 3x the memory on a smaller workload. They ruled out the
    # machine, gradient checkpointing and sequence length individually. Depth
    # through linear attention is the cost.
    #
    # So the honest model is: keep their formula, and multiply the activation
    # term when the base is hybrid. The multiplier is FITTED to this machine's
    # own two measured Qwen3.5-4B runs, not guessed:
    #
    #   seq 128  peak  3.01 GB   (deli-counter-helper)
    #   seq 768  peak 11.61 GB   (instructor-v1, abandoned)
    #
    # 5.0 reproduces the seq-768 point almost exactly and over-predicts the
    # short-sequence one, which is the safe direction: an over-estimate leaves
    # checkpointing on, an under-estimate costs three hours in swap.
    hybrid_penalty = 5.0 if dims.is_hybrid else 1.0

    est = dims.weights_gb + act_gb * hybrid_penalty + 1.0
    if est <= budget_gb * 0.8:
        verdict = "fits"
    elif est <= budget_gb:
        verdict = "tight"
    else:
        verdict = "wont_fit"
    return est, budget_gb, verdict
