#!/usr/bin/env python3
"""Run mlx-vlm's LoRA trainer with the linear-attention layers excluded.

WHY THIS WRAPPER EXISTS
-----------------------
Qwen3.5 is a hybrid model. `full_attention_interval: 4` means only every fourth
layer is classic attention; the other three quarters are GatedDeltaNet
`linear_attn.*`. llama.cpp's Qwen3.5 converter reorders linear-attention V heads
from grouped to tiled order. That reorder is a row permutation, and a row
permutation of W is only valid on the B side of a low-rank factorisation
W ≈ B@A. Propagated to both factors it yields an adapter that loads without
complaint and is quietly wrong — the worst failure mode available here.

Flight School's own text pipeline solves this by restricting mlx-lm's LoRA keys
to `self_attn.*` and `mlp.*`. mlx-vlm has no equivalent flag: it calls
`find_all_linear_names(model.language_model)` and takes everything. So this
script patches that one function before handing control to mlx-vlm's own main.

The alternative — training everything and dropping the unconvertible tensors at
export — would mean the adapter you check-rode is not the adapter you trained.
Excluding them up front keeps those two the same object.

Usage: mlx-vlm's own `python -m mlx_vlm lora` arguments, plus
`--min-pixels N --max-pixels N`: the projector's picture size limits (spec
D80), which replace the processor's own so training sees each picture at the
size llama.cpp serves it (`pigeon/picturesize.py`).
"""

from __future__ import annotations

import sys

# Restricted to the tensors whose GGUF mapping is a plain rename.
ALLOWED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
# Bare module names, because find_all_linear_names returns leaf names on some
# mlx-vlm versions and dotted paths on others. Both spellings are handled.
ALLOWED_LEAVES = {a.split(".")[-1] for a in ALLOWED}


# MEASURED, not guessed. The dog-breed run reported "Peak mem 10.808 GB" against
# an 11.4 GB budget and then died at iter 320 with a Metal OOM — after 300 clean
# iterations. The cause is not per-example sizing: both corpora produce only
# ~150-320 vision tokens, far inside the 1280-token allowance, and the static
# estimate of 7.8 GB was reasonable. What grows over a long run is MLX's Metal
# buffer CACHE, which sits on top of the working set until it crowds it out.
#
# Capping the cache trades a little throughput for a run that finishes. The
# limit is deliberately well under budget so the working set always has room.
CACHE_LIMIT_GB = 2.0


def _take_limits() -> dict | None:
    """--min-pixels/--max-pixels, taken out of argv before mlx-vlm parses it."""
    out = {}
    for flag, key in (("--min-pixels", "min_pixels"), ("--max-pixels", "max_pixels")):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            out[key] = int(sys.argv[i + 1])
            del sys.argv[i:i + 2]
    return out if len(out) == 2 else None


def _apply_limits(processor, lim: dict) -> None:
    """pigeon/picturesize.py apply_to_processor, repeated here because this
    script runs in the vision environment, without the pigeon package."""
    ip = getattr(processor, "image_processor", processor)
    for obj in (ip, processor):
        for k in ("min_pixels", "max_pixels"):
            if hasattr(obj, k):
                setattr(obj, k, lim[k])
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        size = dict(size)
        if "shortest_edge" in size or "longest_edge" in size:
            size["shortest_edge"], size["longest_edge"] = lim["min_pixels"], lim["max_pixels"]
        if "min_pixels" in size or "max_pixels" in size:
            size["min_pixels"], size["max_pixels"] = lim["min_pixels"], lim["max_pixels"]
        ip.size = size


def main() -> int:
    # THE PATCH POINT, and why it is trainer.utils rather than mlx_vlm.lora:
    # runpy.run_module re-executes mlx_vlm.lora in a FRESH namespace, so any
    # attribute set on the already-imported lora module is discarded before it
    # runs (measured: the first attempt trained fully unrestricted and never
    # printed the line below). But that fresh execution does
    # `from .trainer.utils import find_all_linear_names` — which reads the
    # CURRENT attribute of trainer.utils. Patching the source module is what
    # survives the re-import.
    import mlx.core as mx
    mx.set_cache_limit(int(CACHE_LIMIT_GB * 1024 ** 3))
    print(f"[instructor] MLX cache capped at {CACHE_LIMIT_GB} GB "
          f"(a long vision run OOMs at iter ~320 without this)", flush=True)

    import mlx_vlm.trainer.utils as trainer_utils

    original = trainer_utils.find_all_linear_names

    def restricted(model):
        names = list(original(model))
        kept = [n for n in names
                if any(n.endswith(a) for a in ALLOWED)
                or n.split(".")[-1] in ALLOWED_LEAVES]
        dropped = [n for n in names if n not in kept]
        print(f"[instructor] LoRA restricted to attention + MLP projections: "
              f"kept {len(kept)}, excluded {len(dropped)}", flush=True)
        if dropped:
            sample = ", ".join(sorted({d.split('.')[-1] for d in dropped})[:8])
            print(f"[instructor] excluded module types: {sample}", flush=True)
        if not kept:
            print("[instructor] FATAL: nothing left to train after the "
                  "linear-attention exclusion. Refusing rather than training a "
                  "model whose adapter could not be exported.", flush=True)
            raise SystemExit(2)
        return kept

    trainer_utils.find_all_linear_names = restricted

    # D80: the picture size limits come from the base's projector, not the
    # processor's defaults (Qwen3-VL's enlarged every picture to 256 x 256).
    # Patched on mlx_vlm.utils for the same reason as above: lora's fresh
    # execution does `from .utils import load`, reading the current attribute.
    lim = _take_limits()
    if lim:
        import mlx_vlm.utils as vlm_utils
        original_load = vlm_utils.load

        def load_with_limits(*a, **kw):
            model, processor = original_load(*a, **kw)
            _apply_limits(processor, lim)
            print(f"[instructor] picture size limits from the projector: {lim['min_pixels']} to "
                  f"{lim['max_pixels']} pixels", flush=True)
            return model, processor

        vlm_utils.load = load_with_limits

    import runpy
    sys.argv[0] = "mlx_vlm.lora"
    runpy.run_module("mlx_vlm.lora", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
