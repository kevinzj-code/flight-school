"""mlx adapter → PEFT → GGUF, with a gate that refuses a quietly-wrong adapter.

The two vendored converters do the real work and are called, never rewritten.
What this module adds is the part the spec cares about most:

  "The Instructor will produce confident, well-shaped nonsense sometimes."

The same is true of a conversion. An adapter whose factors were transformed
incorrectly still loads, still generates, and is wrong in a way no error message
will ever tell you about. So after conversion, the delta implied by the GGUF is
reconstructed and compared against the delta the MLX adapter implies. Worst
relative error must be ~0, the same check ADAPTERS.md §8 ran by hand.

For vision adapters there is one extra step: mlx-vlm writes language-model
tensors under a `language_model.` / `model.language_model.` prefix, which the
text converter does not recognise. Those names are remapped first, and the gate
above is what proves the remap was right rather than merely plausible.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

# mlx-vlm nests the LLM. Strip the nesting so the tensor names match what the
# text path (and llama.cpp's GGUF naming) expects.
_VLM_PREFIXES = (
    "model.language_model.",
    "language_model.model.",
    "language_model.",
)


class ConversionError(RuntimeError):
    pass


@dataclass
class ConversionResult:
    gguf_path: Path
    peft_dir: Path
    n_tensors: int
    remapped: int = 0
    worst_relative_error: float | None = None
    verified: bool = False
    notes: list[str] = field(default_factory=list)


def remap_vision_adapter(adapter_dir: Path, out_dir: Path) -> tuple[Path, int, int]:
    """Rewrite an mlx-vlm adapter into the mlx-lm shape the converters expect.

    Three differences, all verified against mlx-vlm's own source
    (`trainer/lora_layers.py::LoRaLayer`, `lora.py::setup_model_for_training`):

    1. NAMES. mlx-vlm writes `.A` / `.B`; mlx-lm writes `.lora_a` / `.lora_b`.
       Orientation is identical — A is (in, rank), B is (rank, out) in both — so
       this is a rename, not a transpose.
    2. PREFIX. The LLM sits under `language_model.` / `model.language_model.`.
       llama.cpp's naming has no such nesting, so it is stripped.
    3. SCALE. mlx-vlm's LoRaLayer sets `self.scale = alpha / rank`; mlx-lm
       applies `scale` directly. Writing `scale = alpha/rank` into the config
       makes the downstream `lora_alpha = scale * rank` land back on mlx-vlm's
       own alpha. Getting this wrong does not fail loudly — it produces an
       adapter that is uniformly too strong or too weak.

    Two families of tensor are DROPPED rather than converted:

    - vision tower / projector LoRA. llama.cpp applies a GGUF adapter to the
      language model only; vision goes through the separate mmproj file.
    - `linear_attn.*` LoRA. Qwen3.5 is hybrid, and llama.cpp's converter
      reorders linear-attention V heads. That reorder is only valid on the B
      side of W ≈ B@A; applied to both factors it yields an adapter that loads
      cleanly and is quietly wrong. mlx-vlm has no flag to exclude these
      modules, so `scripts/vlm_lora_restricted.py` excludes them at training
      time and this is the belt-and-braces check.

    Returns (directory, renamed, dropped).
    """
    import json as _json

    from safetensors.numpy import load_file, save_file  # convert venv only

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = None
    for candidate in ("adapters.safetensors", "adapter_model.safetensors"):
        if (adapter_dir / candidate).exists():
            src = adapter_dir / candidate
            break
    if src is None:
        raise ConversionError(f"no adapter safetensors file in {adapter_dir}")

    tensors = load_file(str(src))
    remapped: dict[str, object] = {}
    renamed = dropped = 0
    ranks: set[int] = set()

    for key, value in tensors.items():
        if any(bad in key for bad in ("vision_tower", "vision_model", "visual",
                                      "mm_projector", "multi_modal_projector",
                                      "linear_attn")):
            dropped += 1
            continue
        if key.endswith(".A"):
            stem, suffix = key[:-2], ".lora_a"
        elif key.endswith(".B"):
            stem, suffix = key[:-2], ".lora_b"
        elif key.endswith(".lora_a") or key.endswith(".lora_b"):
            stem, suffix = key.rsplit(".lora_", 1)[0], ".lora_" + key[-1]
        else:
            dropped += 1
            continue
        for prefix in _VLM_PREFIXES:
            if stem.startswith(prefix):
                stem = "model." + stem[len(prefix):]
                renamed += 1
                break
        # mlx-vlm keeps the wrapped layer as `.original_layer`; the LoRA factors
        # hang off the module path itself, so nothing else needs stripping.
        stem = stem.replace(".original_layer", "")
        remapped[stem + suffix] = value
        ranks.add(value.shape[1] if suffix == ".lora_a" else value.shape[0])

    if not remapped:
        raise ConversionError(
            f"after remapping, {src} had no convertible language-model LoRA "
            f"tensors left. Everything in it was vision-tower or linear-attention "
            f"LoRA, neither of which can be represented in a GGUF adapter.")
    if len(ranks) != 1:
        raise ConversionError(
            f"the adapter carries mixed ranks {sorted(ranks)}; converting with a "
            f"single rank would silently mis-scale it.")
    rank = ranks.pop()

    save_file(remapped, str(out_dir / "adapters.safetensors"))

    # Recover mlx-vlm's alpha so the scale conversion is exact rather than
    # assumed. Fall back to mlx-vlm's own CLI default only if nothing recorded it.
    alpha = None
    for name in ("adapter_config.json", "config.json"):
        cfg_file = adapter_dir / name
        if cfg_file.exists():
            try:
                blob = _json.loads(cfg_file.read_text())
            except (OSError, ValueError):
                continue
            alpha = (blob.get("lora_alpha") or blob.get("alpha")
                     or (blob.get("lora") or {}).get("alpha"))
            if alpha:
                break
    if alpha is None:
        alpha = 16.0        # mlx_vlm/lora.py --lora-alpha default
    scale = float(alpha) / rank

    (out_dir / "adapter_config.json").write_text(_json.dumps({
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0},
        "_provenance": "remapped from mlx-vlm by instructor/convert.py",
        "_mlx_vlm_alpha": float(alpha),
    }, indent=2) + "\n")

    if dropped:
        (out_dir / "REMAP-NOTES.txt").write_text(
            f"{renamed} language-model tensor(s) renamed from the mlx-vlm layout.\n"
            f"{dropped} tensor(s) dropped (vision tower and/or linear-attention "
            f"LoRA). llama.cpp applies a GGUF adapter to the language model only, "
            f"and linear-attention LoRA has no correct GGUF representation.\n")
    return out_dir, renamed, dropped


# The verification gate. Runs inside the convert venv, which has gguf, torch and
# safetensors. Returns the worst relative error across every LoRA module.
_VERIFY = r'''
import json, sys, re
# gguf-py is vendored beside the converter, never installed from PyPI (see
# requirements-convert.txt). The converter adds it to its own path; this
# snippet has to do the same, or a fresh Mac's environment cannot import it —
# the second Mac's first conversion failed exactly here, because on the
# machine this was written on someone had pip-installed it by hand once.
sys.path.insert(1, sys.argv[5])
import numpy as np
from gguf import GGUFReader
from safetensors.numpy import load_file

gguf_path, mlx_path, cfg_path, mlx_cfg_path = sys.argv[1:5]

# GGUF does NOT use Hugging Face tensor names. llama.cpp renames every layer to
# its own scheme, so pairing the two sides by string suffix silently matches
# nothing — which reads as "verified" if you only look at the error number.
# This table is the actual mapping, and a module that is not in it is REPORTED
# rather than skipped.
HF_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj":    "ffn_gate",
    "mlp.up_proj":      "ffn_up",
    "mlp.down_proj":    "ffn_down",
}

def to_gguf_name(hf_name):
    m = re.search(r"layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        return None
    layer, module = m.group(1), m.group(2)
    mapped = HF_TO_GGUF.get(module)
    return f"blk.{layer}.{mapped}.weight" if mapped else None

cfg = json.load(open(cfg_path))
rank = int(cfg.get("r") or cfg.get("rank") or 8)
alpha = float(cfg.get("lora_alpha", rank))
peft_scale = alpha / rank

# The mlx-side multiplier, read from the adapter that produced these weights
# rather than assumed. Text runs use 20.0; a remapped mlx-vlm run uses
# alpha/rank. Assuming either one would make this gate agree with itself.
mlx_cfg = json.load(open(mlx_cfg_path))
mlx_scale = float((mlx_cfg.get("lora_parameters") or {}).get("scale", 20.0))

mlx = load_file(mlx_path)
reader = GGUFReader(gguf_path)

g = {}
for t in reader.tensors:
    m = re.match(r"^(.*)\.lora_(a|b)$", t.name)
    if m:
        g.setdefault(m.group(1), {})[m.group(2)] = np.array(t.data, dtype=np.float32)

m_pairs = {}
for k, v in mlx.items():
    m = re.match(r"^(.*)\.lora_(a|b)$", k)
    if m:
        m_pairs.setdefault(m.group(1), {})[m.group(2)] = np.asarray(v, dtype=np.float32)

worst, compared, missing = 0.0, 0, []
for name, parts in sorted(m_pairs.items()):
    if "a" not in parts or "b" not in parts:
        missing.append(f"{name} (incomplete pair)")
        continue
    # mlx: y = x @ a @ b, a is (in, r), b is (r, out)
    #      delta = scale * (b.T @ a.T)  ->  (out, in)
    d_mlx = mlx_scale * (parts["b"].T @ parts["a"].T)

    gname = to_gguf_name(name)
    if gname is None or gname not in g:
        missing.append(f"{name} -> {gname or 'unmapped'}")
        continue
    ga, gb = g[gname].get("a"), g[gname].get("b")
    if ga is None or gb is None:
        missing.append(f"{gname} (incomplete pair)")
        continue

    # PEFT/GGUF orientation: a is (r, in), b is (out, r); delta = scale * b @ a.
    # Candidates are built lazily — evaluating them all eagerly raises on the
    # orientations that do not conform, which is the whole point of trying.
    d_gguf = None
    for build in (lambda: gb @ ga, lambda: (gb @ ga).T,
                  lambda: gb.T @ ga.T, lambda: (gb.T @ ga.T).T):
        try:
            cand = build()
        except ValueError:
            continue
        if cand.shape == d_mlx.shape:
            d_gguf = peft_scale * cand
            break
    if d_gguf is None:
        missing.append(f"{gname} (no orientation matches {d_mlx.shape})")
        continue

    denom = np.abs(d_mlx).max() or 1.0
    worst = max(worst, float(np.abs(d_gguf - d_mlx).max() / denom))
    compared += 1

print(json.dumps({"worst": worst, "compared": compared,
                  "missing": missing[:8], "n_missing": len(missing)}))
'''



def verify_roundtrip(cfg: Config, gguf_path: Path, mlx_adapter: Path,
                     peft_dir: Path) -> tuple[float, int, list[str]]:
    """Compare the GGUF's implied delta against the MLX adapter's own."""
    cfg_json = peft_dir / "adapter_config.json"
    proc = subprocess.run(
        [str(cfg.venv_convert), "-c", _VERIFY, str(gguf_path),
         str(mlx_adapter / "adapters.safetensors"), str(cfg_json),
         str(mlx_adapter / "adapter_config.json"),
         str(Path(cfg.lora_to_gguf).parent / "gguf-py")],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise ConversionError(
            f"verification could not run:\n{(proc.stderr or '')[-600:]}")
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ConversionError(
            f"verification produced no verdict: {proc.stdout[:300]}") from exc
    return data["worst"], data["compared"], data.get("missing", [])


def _base_snapshot(cfg: Config, repo: str) -> Path:
    """Local snapshot of the base's *config* files, so the converter reads them
    off disk instead of reaching for the network.

    The cache is tried FIRST, on its own. Plain snapshot_download contacts the
    Hub to resolve the revision on every call even when everything it needs is
    already on disk — a round-trip per conversion, and a stall rather than a
    conversion when the machine is offline, in an app whose whole promise is
    that it works without a network. The networked call remains the fallback
    for a base that genuinely is not cached yet.

    The local answer is only accepted when config.json is actually in it: a
    snapshot cached earlier under narrower allow_patterns can come back
    present-but-incomplete, and the converter would then fail somewhere much
    less obvious than here.
    """
    code = (
        "import os, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "repo, pats = sys.argv[1], ['*.json']\n"
        "cached = None\n"
        "try:\n"
        "    p = snapshot_download(repo, allow_patterns=pats, local_files_only=True)\n"
        "    if os.path.exists(os.path.join(p, 'config.json')):\n"
        "        cached = p\n"
        "except Exception:\n"
        "    pass\n"
        "print(cached or snapshot_download(repo, allow_patterns=pats))\n"
    )
    proc = subprocess.run(
        [str(cfg.venv_convert), "-c", code, repo],
        capture_output=True, text=True, env=cfg.train_env(), timeout=900,
    )
    if proc.returncode != 0:
        raise ConversionError(f"could not resolve base config for {repo}:\n{proc.stderr[-400:]}")
    return Path(proc.stdout.strip().splitlines()[-1])


def to_gguf(
    cfg: Config,
    *,
    adapter_dir: Path,
    out_dir: Path,
    repo: str,
    name: str,
    vision: bool = False,
    verify: bool = True,
    # Calibrated to f16 GGUF storage, NOT plucked from the air. The adapter is
    # written with --outtype f16, whose relative precision is ~1e-3, while the
    # MLX side is f32. A correct conversion therefore lands around 4e-4, not at
    # zero — measured 4.5e-04 across all 32 modules of the Step 0 adapter.
    # ADAPTERS.md §8 reports 0.0 because it compared f32 against f32, before
    # the GGUF write. Tightening this below f16 precision would fail every
    # correct conversion; loosening it much past 1e-3 would stop catching a
    # genuinely mis-transformed factor.
    max_relative_error: float = 1e-3,
) -> ConversionResult:
    """The whole path, with the gate at the end."""
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    n_remapped = 0

    source_dir = adapter_dir
    if vision:
        code = (
            "import sys, json\n"
            "sys.path.insert(0, sys.argv[3])\n"
            "from instructor.convert import remap_vision_adapter\n"
            "d, n, drop = remap_vision_adapter(sys.argv[1], sys.argv[2])\n"
            "print(json.dumps({'dir': str(d), 'renamed': n, 'dropped': drop}))\n"
        )
        remap_dir = out_dir / "adapters-remapped"
        proc = subprocess.run(
            [str(cfg.venv_convert), "-c", code, str(adapter_dir), str(remap_dir),
             str(Path(__file__).resolve().parent.parent)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise ConversionError(f"vision remap failed:\n{(proc.stderr or '')[-600:]}")
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        source_dir = Path(info["dir"])
        n_remapped = info["renamed"]
        notes.append(f"{n_remapped} language-model tensor(s) renamed from the "
                     f"mlx-vlm layout")
        if info.get("dropped"):
            notes.append(
                f"{info['dropped']} tensor(s) dropped as unconvertible "
                f"(vision tower and/or linear-attention LoRA)")

    # 1. mlx → PEFT
    peft_dir = out_dir / "peft"
    proc = subprocess.run(
        [str(cfg.venv_convert), str(cfg.mlx_to_peft),
         "--adapter-dir", str(source_dir),
         "--out-dir", str(peft_dir),
         "--base-model", repo,
         "--adapter-name", name],
        capture_output=True, text=True, env=cfg.train_env(), timeout=1800,
    )
    if proc.returncode != 0:
        raise ConversionError(
            f"mlx → PEFT failed:\n{(proc.stdout + proc.stderr)[-800:]}")

    # 2. PEFT → GGUF
    base_dir = _base_snapshot(cfg, repo)
    gguf_path = out_dir / f"{name}.lora.gguf"
    proc = subprocess.run(
        [str(cfg.venv_convert), str(cfg.lora_to_gguf), str(peft_dir),
         "--base", str(base_dir), "--outtype", "f16", "--outfile", str(gguf_path)],
        capture_output=True, text=True, env=cfg.train_env(), timeout=1800,
    )
    if proc.returncode != 0:
        raise ConversionError(
            f"PEFT → GGUF failed:\n{(proc.stdout + proc.stderr)[-800:]}")
    if not gguf_path.exists():
        raise ConversionError(f"converter reported success but {gguf_path} is missing")

    result = ConversionResult(gguf_path=gguf_path, peft_dir=peft_dir,
                              n_tensors=0, remapped=n_remapped, notes=notes)

    # 3. the gate
    if verify:
        worst, compared, missing = verify_roundtrip(cfg, gguf_path, source_dir, peft_dir)
        result.worst_relative_error = worst
        result.n_tensors = compared
        if compared == 0:
            raise ConversionError(
                "verification compared zero modules — the GGUF and the MLX "
                "adapter share no recognisable tensor names, so nothing about "
                "this adapter has been proven. Refusing it.")
        if worst > max_relative_error:
            raise ConversionError(
                f"verification FAILED: worst relative error {worst:.3e} across "
                f"{compared} modules (limit {max_relative_error:.0e}).\n"
                f"This adapter would load cleanly and be wrong. Refusing it."
                + (f"\nUnmatched: {', '.join(missing)}" if missing else ""))
        if missing:
            # Coverage matters as much as the error figure: a gate that checked
            # 2 of 64 modules and reported 0.0 has proven almost nothing.
            if compared < len(missing):
                raise ConversionError(
                    f"verification covered only {compared} module(s) while "
                    f"{len(missing)} could not be paired. That is not enough "
                    f"coverage to call this adapter proven.\n"
                    f"Unmatched: {', '.join(missing[:4])}")
            result.notes.append(
                f"{len(missing)} module(s) had no GGUF counterpart and were not "
                f"checked: {', '.join(missing[:4])}")
        result.verified = True
    return result
