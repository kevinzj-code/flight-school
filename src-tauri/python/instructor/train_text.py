"""Text training: mlx-lm QLoRA, run as a subprocess.

Every hyperparameter here is frozen, taken from Flight School's training.rs,
which tuned them once and wrote down why. The Instructor picks a preset — time
and depth — and nothing else. It never sees a learning rate.

Two things in this file are load-bearing and must not be "tidied":

1. THE LoRA KEYS. Qwen3.5 is a HYBRID model: `full_attention_interval: 4` means
   only every 4th layer is classic attention; the rest are GatedDeltaNet
   `linear_attn.*`. llama.cpp's converter reorders linear-attention V heads, and
   pushing that reorder through a low-rank factorisation is only valid on the B
   side of W ≈ B@A. A wrapper that applies it to both factors yields an adapter
   that LOADS CLEANLY AND IS QUIETLY WRONG. Restricting LoRA to self_attn.* and
   mlp.* costs some capacity and buys a conversion path that is a plain rename.

2. EARLY STOPPING. A 107-row Flight School run bottomed out its validation loss
   at iter 100 and then rose for 200 more iterations while train loss fell to
   0.21 — textbook overfitting that "finished successfully". Three consecutive
   rises and this stops the run.
"""

from __future__ import annotations

import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config
from .planspec import PRESETS

# See note 1. Restricted to tensors whose GGUF mapping is a plain rename.
QWEN35_LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

STEPS_PER_EVAL = 50
SAVE_EVERY = 100
EARLY_STOP_RISES = 3

_ITER = re.compile(r"^Iter (\d+):")


@dataclass
class TrainProgress:
    iter: int
    total: int
    kind: str            # "train" | "val"
    loss: float | None
    its: float | None = None       # iterations a second, as mlx-lm reports it
    peak_gb: float | None = None   # peak memory so far, as mlx-lm reports it


@dataclass
class TrainResult:
    adapter_dir: Path
    log_path: Path
    iters_run: int
    train_losses: list[tuple[int, float]] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    early_stopped: bool = False
    returncode: int = 0

    @property
    def best_val(self) -> tuple[int, float] | None:
        return min(self.val_losses, key=lambda t: t[1]) if self.val_losses else None


def parse_progress(line: str, total: int) -> TrainProgress | None:
    """mlx-lm's `Iter N: Train loss X, ...` / `Iter N: Val loss X, ...`."""
    m = _ITER.match(line.strip())
    if not m:
        return None
    it = int(m.group(1))
    rest = line.split(":", 1)[1]
    kind = "val" if "Val loss" in rest else "train" if "Train loss" in rest else None
    if kind is None:
        return None
    lm = re.search(r"(?:Val|Train) loss ([0-9.]+)", rest)
    its = re.search(r"It/sec ([0-9.]+)", rest)
    peak = re.search(r"Peak mem ([0-9.]+) GB", rest)
    return TrainProgress(it, total, kind, float(lm.group(1)) if lm else None,
                         float(its.group(1)) if its else None, float(peak.group(1)) if peak else None)


def write_lora_config(path: Path, rank: int = 8, keys: list[str] | None = None) -> Path:
    """mlx-lm reads rank and keys ONLY from this file — there is no CLI flag."""
    lines = ["lora_parameters:", f"  rank: {rank}", "  dropout: 0.0", "  scale: 20.0"]
    if keys:
        lines.append("  keys:")
        lines += [f'    - "{k}"' for k in keys]
    path.write_text("\n".join(lines) + "\n")
    return path


def train(
    cfg: Config,
    *,
    data_dir: Path,
    adapter_dir: Path,
    repo: str,
    preset: str,
    max_seq_length: int,
    grad_checkpoint: bool = True,
    rank: int = 8,
    num_layers: int = 8,
    learning_rate: str = "1e-5",
    log_path: Path | None = None,
    on_progress: Callable[[TrainProgress], None] | None = None,
    restrict_keys: bool = True,
) -> TrainResult:
    """Run mlx-lm and watch it. The caller must already hold the engine lock."""
    if not cfg.venv_train.exists():
        raise RuntimeError(f"training venv missing at {cfg.venv_train}")
    if max_seq_length <= 0 or max_seq_length % 128 != 0:
        raise ValueError("max_seq_length must come from the preflight (multiple of 128)")
    iters = PRESETS[preset]
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or adapter_dir.parent / "train.log"

    config_path = adapter_dir.parent / "lora_config.yaml"
    write_lora_config(config_path, rank=rank,
                      keys=QWEN35_LORA_KEYS if restrict_keys else None)

    args = [
        str(cfg.venv_train), "-m", "mlx_lm", "lora",
        "--model", repo,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "-c", str(config_path),
        "--iters", str(iters),
        "--batch-size", "1",
        "--num-layers", str(num_layers),
        "--learning-rate", learning_rate,
        "--max-seq-length", str(max_seq_length),
        "--steps-per-eval", str(STEPS_PER_EVAL),
        "--save-every", str(SAVE_EVERY),
        "--fine-tune-type", "lora",
        # Loss over the ANSWER only. Every row would otherwise spend capacity
        # re-predicting a question the model is always handed.
        "--mask-prompt",
        "--seed", "11",
    ]
    if grad_checkpoint:
        args.append("--grad-checkpoint")

    result = TrainResult(adapter_dir=adapter_dir, log_path=log_path, iters_run=0)
    rises = 0

    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=cfg.train_env(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            p = parse_progress(line, iters)
            if p is None:
                continue
            result.iters_run = max(result.iters_run, p.iter)
            if p.loss is not None:
                (result.val_losses if p.kind == "val" else result.train_losses).append(
                    (p.iter, p.loss))
            if on_progress:
                on_progress(p)
            if p.kind == "val" and p.loss is not None and len(result.val_losses) >= 2:
                rises = rises + 1 if result.val_losses[-1][1] > result.val_losses[-2][1] else 0
                if rises >= EARLY_STOP_RISES:
                    result.early_stopped = True
                    log_fh.write(
                        f"\n[instructor] validation loss rose {EARLY_STOP_RISES} times "
                        f"in a row — stopping early and keeping the best checkpoint.\n")
                    proc.send_signal(signal.SIGINT)
                    break
        proc.stdout.close()
        result.returncode = proc.wait()

    if result.early_stopped:
        # An interrupted mlx-lm leaves adapters.safetensors as the LAST save,
        # which by construction is after the validation minimum — early stop
        # only fires once loss has been rising. "Keeping the best checkpoint"
        # has to be made true here: find the numbered save closest to (and not
        # after) the best-val iteration and promote it.
        result.returncode = 0
        if result.best_val:
            best_iter = result.best_val[0]
            saves = sorted(
                (int(f.name.split("_")[0]), f)
                for f in adapter_dir.glob("*_adapters.safetensors")
            )
            candidates = [(i, f) for i, f in saves if i <= best_iter] or saves[:1]
            if candidates:
                chosen_iter, chosen = candidates[-1]
                import shutil as _sh
                _sh.copy2(chosen, adapter_dir / "adapters.safetensors")
                with log_path.open("a") as fh:
                    fh.write(f"[instructor] promoted checkpoint {chosen.name} "
                             f"(iter {chosen_iter}, nearest save at or before the "
                             f"best validation iter {best_iter}) to "
                             f"adapters.safetensors.\n")
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        raise RuntimeError(f"mlx-lm exited {result.returncode}:\n{tail}")
    if not (adapter_dir / "adapters.safetensors").exists():
        raise RuntimeError(
            f"training finished but {adapter_dir/'adapters.safetensors'} is missing")
    return result


def ensure_weights(cfg: Config, repo: str,
                   on_line: Callable[[str], None] | None = None) -> None:
    """Fetch the MLX weights if they are not cached. Says what it is doing."""
    from .config import _mlx_cached
    if _mlx_cached(cfg, repo):
        return
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "p = snapshot_download(sys.argv[1])\n"
        "print(p)\n"
    )
    proc = subprocess.Popen(
        [str(cfg.venv_train), "-c", code, repo],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, env=cfg.train_env(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_line:
            on_line(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"could not download {repo}")
