"""Vision training: mlx-vlm QLoRA, run as a subprocess.

mlx-lm cannot do this — its Qwen3.5 loader drops every `vision_tower.*` and
`model.visual.*` key by design (mlx_lm/models/qwen3_5.py). So vision goes
through mlx-vlm, which trains LoRA on the language model with the vision tower
frozen. That is the right architecture *and* the only one compatible with a
GGUF adapter, because llama.cpp handles vision through the separate mmproj file.

The LoRA module restriction is applied by scripts/vlm_lora_restricted.py rather
than here — see that file for why it cannot be a config change.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config, ROOT
from .planspec import PRESETS

def _wrapper_path() -> Path:
    """The LoRA-restriction wrapper, wherever this package is living.

    Beside the package (`<pkg>/../scripts/`) when it is vendored into an app
    bundle, and ROOT/scripts when it is running from this repo's own checkout.
    Resolved rather than assumed because the two layouts are both real and the
    failure — mlx-vlm training every layer including the ones that cannot be
    converted safely — is silent."""
    here = Path(__file__).resolve().parent.parent / "scripts" / "vlm_lora_restricted.py"
    if here.is_file():
        return here
    return ROOT / "scripts" / "vlm_lora_restricted.py"


WRAPPER = _wrapper_path()

# mlx-vlm reports differently from mlx-lm.
_STEP = re.compile(r"(?:Iter|Step)\s+(\d+)[:\s]", re.I)
_LOSS = re.compile(r"(?:train[_ ]?loss|loss)[\s:=]+([0-9.]+)", re.I)
_VAL = re.compile(r"val(?:idation)?[_ ]?loss[\s:=]+([0-9.]+)", re.I)
# mlx-vlm colours its numbers; the colour codes sit between "loss" and the
# value, so they are taken out before anything is matched.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ITS = re.compile(r"It/sec\s+([0-9.]+)")
_PEAK = re.compile(r"Peak mem\s+([0-9.]+)\s*GB")


@dataclass
class VisionTrainResult:
    adapter_dir: Path
    log_path: Path
    iters_run: int = 0
    train_losses: list[tuple[int, float]] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    returncode: int = 0
    #: Set when the trainer died mid-run and a periodic checkpoint was promoted
    #: in place of the final weights. The iteration it came from, or None.
    salvaged_from_iter: int | None = None


def _limit_args(cfg: Config) -> list[str]:
    mmproj = getattr(cfg, "target_mmproj", None)
    if not mmproj:
        return []
    from pigeon.picturesize import limits
    lim = limits(mmproj)
    return ["--min-pixels", str(lim["min_pixels"]), "--max-pixels", str(lim["max_pixels"])]


def available(cfg: Config) -> bool:
    return cfg.venv_vision.exists()


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
    lora_alpha: float = 16.0,
    learning_rate: str = "1e-5",
    batch_size: int = 1,
    log_path: Path | None = None,
    on_progress: Callable[[int, int, str, float | None], None] | None = None,
    on_speed: Callable[[float | None, float | None], None] | None = None,
) -> VisionTrainResult:
    """Run the restricted mlx-vlm trainer. Caller must hold the engine lock."""
    if not available(cfg):
        raise RuntimeError(
            "the vision environment is not set up — Quick build offers to build it "
            "when a plan needs images")
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or adapter_dir.parent / "train.log"
    iters = PRESETS[preset]

    args = [
        str(cfg.venv_vision), str(WRAPPER),
        "--model-path", repo,
        # A directory of train.jsonl / valid.jsonl; datasets' folder builder
        # picks up the split names from the file stems.
        "--dataset", str(data_dir),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", learning_rate,
        "--max-seq-length", str(max_seq_length),
        "--lora-rank", str(rank),
        "--lora-alpha", str(lora_alpha),
        "--steps-per-report", "10",
        # INERT, and passed only so the command line stays honest about what was
        # asked for. mlx-vlm 0.7.0's lora.py hardcodes `val_dataset=None` at both
        # of its training call sites (lines 231 and 256), so no validation set can
        # reach the trainer through this CLI and `--steps-per-eval` has nothing to
        # evaluate. Measured: a run whose data directory contained a 13-row
        # valid.jsonl still logged "No validation dataset provided".
        #
        # Consequences, both real: there is no validation signal on the vision
        # path, so nothing can early-stop it; and the rows Flight School withholds
        # as a validation split are neither trained on nor evaluated. See
        # the baseline notes, Phase 2.
        "--steps-per-eval", "50",
        # This one is NOT inert, and it is what makes a late crash survivable —
        # the promoted checkpoint comes from here.
        "--steps-per-save", "100",
        # Loss over the answer only, the same choice --mask-prompt makes on the
        # text path.
        "--train-on-completions",
        "--output-path", str(adapter_dir / "adapters.safetensors"),
        # D80: the projector's picture size limits, so training sees each
        # picture at the size llama.cpp serves it (pigeon/picturesize.py).
        *_limit_args(cfg),
    ]
    if grad_checkpoint:
        args.append("--grad-checkpoint")

    result = VisionTrainResult(adapter_dir=adapter_dir, log_path=log_path)
    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=cfg.train_env())
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            line = _ANSI.sub("", line)
            m = _STEP.search(line)
            if not m:
                continue
            step = int(m.group(1))
            result.iters_run = max(result.iters_run, step)
            vm = _VAL.search(line)
            lm = _LOSS.search(line)
            if vm:
                result.val_losses.append((step, float(vm.group(1))))
                kind, loss = "val", float(vm.group(1))
            elif lm:
                result.train_losses.append((step, float(lm.group(1))))
                kind, loss = "train", float(lm.group(1))
            else:
                kind, loss = "train", None
            if on_speed:
                its, peak = _ITS.search(line), _PEAK.search(line)
                on_speed(float(its.group(1)) if its else None, float(peak.group(1)) if peak else None)
            if on_progress:
                on_progress(step, iters, kind, loss)
        proc.stdout.close()
        result.returncode = proc.wait()

    final = adapter_dir / "adapters.safetensors"

    if result.returncode != 0:
        # A VISION RUN THAT DIES LATE IS NOT A WASTED RUN. These converge early
        # — the pilot's dog corpus was at 0.0007 train loss by iter 300 — and
        # the failure mode they die of is a Metal out-of-memory that arrives
        # after the useful training is done. The pilot hit exactly this at iter
        # 320 of 600, and a person rescued the iter-300 checkpoint by hand; the
        # cartridge then flew 36/36. Doing that by hand is not a workflow.
        #
        # So: promote the newest periodic checkpoint and carry on, loudly. This
        # does not weaken "the test decides which checkpoint ships" — the check
        # ride still runs on whatever is promoted, and it is the arbiter. What
        # it prevents is throwing away a finished-enough adapter because the
        # process exited non-zero after producing it.
        salvage = _newest_checkpoint(adapter_dir)
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        if salvage is None:
            raise RuntimeError(f"mlx-vlm exited {result.returncode}:\n{tail}")
        step, path = salvage
        shutil.copy2(path, final)
        result.salvaged_from_iter = step
        return result

    if not final.exists():
        raise RuntimeError(
            f"training finished but {final} is missing")
    return result


_CHECKPOINT = re.compile(r"^(\d+)_adapters\.safetensors$")


def _newest_checkpoint(adapter_dir: Path) -> tuple[int, Path] | None:
    """The highest-numbered periodic checkpoint mlx-vlm wrote, if any.

    Written every `--steps-per-save` iterations as `0000300_adapters.safetensors`.
    """
    best: tuple[int, Path] | None = None
    for entry in Path(adapter_dir).iterdir():
        m = _CHECKPOINT.match(entry.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, entry)
    return best
