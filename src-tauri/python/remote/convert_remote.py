#!/usr/bin/env python3
"""On the Mac, after a remote run: the pod's adapter to `<name>.lora.gguf` along
the same path as a Mac-trained one (instructor.convert.to_gguf: mlx -> PEFT ->
llama.cpp's convert_lora_to_gguf.py against the base the Pigeon names, then the
round-trip gate), and a check that the GGUF carries the same tensor names,
shapes and types as an MLX-made adapter's.

    convert_remote.py <run_dir> <name> --config CONFIG.toml --reference MLX.lora.gguf

Runs in the converter venv. CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(1, str(HERE.parent / "vendor" / "gguf-py"))

MAC_REPO = "mlx-community/Qwen3-VL-2B-Instruct-bf16"


def tensor_table(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    from gguf import GGUFReader
    return sorted((t.name, tuple(int(x) for x in t.shape), t.tensor_type.name) for t in GGUFReader(str(path)).tensors)


def same_shape(made: list, reference: list) -> list[str]:
    """What differs between two adapters' tensor tables, empty when they agree."""
    a, b = {n: (s, ty) for n, s, ty in made}, {n: (s, ty) for n, s, ty in reference}
    out = [f"missing {n}" for n in sorted(b.keys() - a.keys())]
    out += [f"extra {n}" for n in sorted(a.keys() - b.keys())]
    out += [f"{n}: {a[n]} against {b[n]}" for n in sorted(a.keys() & b.keys()) if a[n] != b[n]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("name")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="an MLX-made .lora.gguf of the same base")
    a = ap.parse_args(argv)

    from instructor import config, convert
    res = convert.to_gguf(config.load(a.config), adapter_dir=a.run_dir / "adapters", out_dir=a.run_dir / "export",
                          repo=MAC_REPO, name=a.name, vision=False)
    diff = same_shape(tensor_table(res.gguf_path), tensor_table(a.reference))
    print(json.dumps({"gguf": str(res.gguf_path), "modules_compared": res.n_tensors,
                      "worst_relative_error": res.worst_relative_error, "reference": str(a.reference),
                      "tensor_table_differences": diff[:20], "matches_reference": not diff}, indent=1))
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
