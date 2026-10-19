"""The second check ride: the same exam, asked in MLX, where the adapter trained.

The check ride proper flies llama.cpp, the engine every host serves with, and
its number is the one a host shows. This one exists to catch what that number
alone cannot: an adapter that scores well where it trained and loses it when
served. Measured on Seed Run (base bake-off, 2026-10-02): a picture Pigeon
trained on 4-bit weights chose the right key 169 times in 200 in MLX and 139
in llama.cpp, and a reasoning one 166 against 4. The conversion gate passed on
every one of them. Spec §12.1 (`check_rides`, D21, D30).

Same rows, same prompt text (fly.ride_user_texts), same grading (grade.py and
the Runner rules in pigeon/runner.py). No grammar — MLX has none to match
llama.cpp's — and no control pass: the question here is training against
serving, not adapter against base.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import grade as grade_mod
from .fly import _norm, label_set, ride_user_texts
from pigeon.pack import serving_trails  # noqa: F401  (quickbuild imports it from here)


def run_mlx_ride(cfg, run_dir: Path, plan: dict, holdout: Sequence[dict[str, Any]], *,
                 max_answer_tokens: int, repo: str, chunks: list[dict] | None = None,
                 precision: str | None = None) -> tuple[dict | None, str | None]:
    """(check_rides entry, None) or (None, why it was not flown)."""
    if grade_mod.needs_judge(plan):
        return None, "answers are graded by a model; the second ride covers exactly-checked answers only"
    vision = any(h.get("image") for h in holdout)
    venv = cfg.venv_vision if vision else cfg.venv_train
    adapters = Path(run_dir) / "adapters"
    if not Path(venv).exists() or not (adapters / "adapters.safetensors").exists():
        return None, "no MLX environment or trained adapter on disk"
    rows = [{"text": t, **({"image": h["image"]} if h.get("image") else {})}
            for t, h in zip(ride_user_texts(plan, holdout, chunks), holdout)]
    budget = max_answer_tokens     # the llama.cpp ride's: the file's output_contract.max_tokens
    req = {"mode": "picture" if vision else "text", "repo": repo, "hf_home": str(cfg.hf_home),
           "adapter_dir": str(adapters), "rows": rows, "max_tokens": budget}
    if vision and getattr(cfg, "target_mmproj", None):
        from pigeon.picturesize import limits
        req["limits"] = limits(cfg.target_mmproj)        # D80: the size training and serving use
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(req, fh)
    script = Path(__file__).with_name("mlx_ask.py")
    try:
        out = subprocess.run([str(venv), str(script), fh.name], capture_output=True, text=True,
                             env=cfg.train_env(), timeout=7200)
        got = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as e:
        return None, f"the MLX ride did not run ({e})"
    finally:
        Path(fh.name).unlink(missing_ok=True)
    if "answers" not in got:
        return None, got.get("error") or "the MLX ride returned nothing"
    labels, rule = label_set(plan), grade_mod.rule_for(plan)
    passed = sum(1 for h, a in zip(holdout, got["answers"])
                 if grade_mod.grade_row(h["answer"], a.strip(), labels=labels, rule=rule,
                                        judge=False, norm=_norm)[0])
    entry = {"runtime": "mlx", "passed": passed, "total": len(holdout), "control_passed": None,
             "temperature": 0, "constrained": False}
    if precision:
        entry["precision"] = precision
    return entry, None
