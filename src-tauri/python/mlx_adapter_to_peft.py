#!/usr/bin/env python3
"""Translate an mlx-lm LoRA adapter into the Hugging Face PEFT layout.

llama.cpp's convert_lora_to_gguf.py only reads PEFT adapters. mlx-lm writes its own
shape, so nothing can go straight from a training run to a GGUF adapter — this script
is the missing link, and it is the only place in Flight School where the two
conventions are reconciled.

Everything below was verified against the installed mlx-lm 0.31.3 (tuner/lora.py) and
a real completed run's adapters.safetensors, not inferred from documentation.

WHAT ACTUALLY DIFFERS
---------------------
1. Orientation. mlx-lm's LoRALinear.__call__ is

       y = W x + scale * ((x @ lora_a) @ lora_b)

   with lora_a shaped (in_features, r) and lora_b shaped (r, out_features). PEFT
   computes the same delta as

       y = W x + (lora_alpha / r) * lora_B @ (lora_A x)

   with lora_A shaped (r, in_features) and lora_B shaped (out_features, r). Since
   `x @ lora_a` is `lora_a.T x`, the mapping is a plain transpose of each:

       lora_A = lora_a.T        lora_B = lora_b.T

   mlx-lm's own fuse() agrees — it builds `delta = (scale * lora_b.T) @ lora_a.T`,
   which is exactly `scale * lora_B @ lora_A`.

2. Scaling. mlx-lm applies `scale` directly. PEFT applies `lora_alpha / r`. Equating
   them gives the conversion this script performs:

       lora_alpha = scale * rank

   Getting this wrong does not fail loudly — it produces an adapter whose effect is
   uniformly too strong or too weak, which reads as "the training sort of worked".
   With Flight School's defaults (scale 20.0, rank 8) the correct alpha is 160.

3. Naming. mlx-lm writes `model.layers.20.self_attn.q_proj.lora_a`. PEFT writes
   `base_model.model.model.layers.20.self_attn.q_proj.lora_A.weight`. The converter's
   get_base_tensor_name() strips the `base_model.model.` prefix and the
   `.lora_A.weight` suffix to recover the base tensor name, so both parts are load-
   bearing and neither is cosmetic.

4. Partial coverage is normal. mlx-lm's `num_layers` defaults to applying LoRA to only
   the LAST N layers (16 of 36 on Qwen3-4B). PEFT and GGUF both handle a sparse
   adapter fine; the untouched layers simply carry no delta.

PROVENANCE
----------
The README.md written beside the adapter is not documentation — gguf-py's
Metadata.load() parses its YAML frontmatter, so the `base_model:` line becomes a real
`general.base_model.0.repo_url` key inside the finished GGUF. That is what lets
Airoplane Mode refuse an adapter built for a different base instead of loading it and
producing noise. See ADAPTERS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

# mlx-lm's own default in tuner/lora.py, used only if a run's config somehow omits it.
DEFAULT_SCALE = 20.0
DEFAULT_RANK = 8


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def peft_key(mlx_key: str) -> tuple[str, bool]:
    """`model.layers.0.self_attn.q_proj.lora_a` ->
    `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`.

    Returns the new key and whether it was the A side, so the caller knows this was a
    LoRA tensor at all rather than something unexpected sitting in the file.
    """
    if mlx_key.endswith(".lora_a"):
        return f"base_model.model.{mlx_key[: -len('.lora_a')]}.lora_A.weight", True
    if mlx_key.endswith(".lora_b"):
        return f"base_model.model.{mlx_key[: -len('.lora_b')]}.lora_B.weight", False
    die(
        f"tensor {mlx_key!r} is neither .lora_a nor .lora_b. This adapter was probably "
        f"not trained with fine_tune_type='lora' — DoRA and full fine-tunes cannot be "
        f"exported as LoRA adapters."
    )
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", type=Path, required=True,
                    help="mlx-lm --adapter-path directory (adapters.safetensors + adapter_config.json)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to write the PEFT-shaped adapter")
    ap.add_argument("--base-model", type=str, required=True,
                    help="base model repo id, recorded for provenance and base matching")
    ap.add_argument("--adapter-name", type=str, default=None,
                    help="human name recorded in the model card (defaults to the out-dir name)")
    args = ap.parse_args()

    weights_path = args.adapter_dir / "adapters.safetensors"
    config_path = args.adapter_dir / "adapter_config.json"
    if not weights_path.exists():
        die(f"{weights_path} not found — has this run finished training?")
    if not config_path.exists():
        die(f"{config_path} not found — this does not look like an mlx-lm adapter directory.")

    mlx_config = json.loads(config_path.read_text())

    fine_tune_type = mlx_config.get("fine_tune_type", "lora")
    if fine_tune_type != "lora":
        die(
            f"fine_tune_type is {fine_tune_type!r}, not 'lora'. Only plain LoRA adapters "
            f"can be converted to GGUF; a {fine_tune_type} run has to be fused and "
            f"exported as a full model instead."
        )

    lora_params = mlx_config.get("lora_parameters") or {}
    rank = int(lora_params.get("rank", DEFAULT_RANK))
    scale = float(lora_params.get("scale", DEFAULT_SCALE))
    # See note 2 above. mlx-lm's `scale` IS PEFT's `lora_alpha / r`.
    lora_alpha = scale * rank

    tensors = load_file(str(weights_path))
    if not tensors:
        die("the adapter file has no tensors in it.")

    out_tensors: dict[str, np.ndarray] = {}
    target_modules: set[str] = set()
    ranks_seen: set[int] = set()

    for key, value in sorted(tensors.items()):
        new_key, is_a = peft_key(key)
        if value.ndim != 2:
            die(f"expected a 2-D tensor for {key}, got shape {value.shape}")
        # np.ascontiguousarray because safetensors will not serialise a bare transposed
        # view — it needs the data laid out in the order the header claims.
        out_tensors[new_key] = np.ascontiguousarray(value.T)
        ranks_seen.add(value.shape[1] if is_a else value.shape[0])
        # `model.layers.20.self_attn.q_proj.lora_a` -> `q_proj`
        parts = key.split(".")
        if len(parts) >= 2:
            target_modules.add(parts[-2])

    if ranks_seen != {rank}:
        die(
            f"adapter_config.json says rank {rank}, but the weights carry rank(s) "
            f"{sorted(ranks_seen)}. Converting with the wrong rank silently mis-scales "
            f"the adapter, so this is refused rather than guessed."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(args.out_dir / "adapter_model.safetensors"))

    peft_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": args.base_model,
        "r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": float(lora_params.get("dropout", 0.0)),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": sorted(target_modules),
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(peft_config, indent=2) + "\n")

    # Parsed by gguf-py's Metadata.load() — this frontmatter is how the base model id
    # reaches the finished GGUF's KV store. Plain prose below it is ignored.
    name = args.adapter_name or args.out_dir.name
    (args.out_dir / "README.md").write_text(
        "---\n"
        f"base_model: {args.base_model}\n"
        f"model_name: {name}\n"
        "library_name: peft\n"
        "tags:\n"
        "  - lora\n"
        "  - airoplane-mode\n"
        "---\n\n"
        f"LoRA adapter `{name}`, trained in Airoplane Mode Flight School on "
        f"`{args.base_model}`.\n\n"
        f"Rank {rank}, alpha {lora_alpha:g} (mlx-lm scale {scale:g} x rank {rank}).\n"
    )

    n_layers = len({k.split(".layers.")[1].split(".")[0] for k in tensors if ".layers." in k})
    print(
        f"converted {len(tensors)} mlx tensors -> {len(out_tensors)} PEFT tensors "
        f"across {n_layers} layers"
    )
    print(f"rank {rank}, mlx scale {scale:g} -> lora_alpha {lora_alpha:g}")
    print(f"target modules: {', '.join(sorted(target_modules))}")
    print(f"base model: {args.base_model}")


if __name__ == "__main__":
    main()
