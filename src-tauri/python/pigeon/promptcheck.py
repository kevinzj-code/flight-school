#!/usr/bin/env python3
"""`prompt_check` (spec §8.6): one input and the exact prompt training built from it.

A Runner rebuilds the prompt for `input` by §8.1–§8.3 and compares it, byte for
byte, with `canonical`. That only proves anything if `canonical` comes from the
code that built the TRAINING rows, so this module renders it with the trainer's
own chat-template path, not with anything a Runner would use:

  * text and grounded — mlx-lm's tokenizer wrapper (`mlx_lm lora` renders every
    chat row through it), in the training venv;
  * pictures — mlx-vlm's `prompt_utils.apply_chat_template` and `prepare_inputs`
    (what `trainer/datasets.py::VisionDataset` does), in the vision venv.

The training row is rendered WITH an answer — a sentinel — and cut where the
answer starts, so whatever a template puts between the user turn and the answer
(an empty reasoning block, a default system prompt) is exactly what training saw.

Pictures are written in canonical form (D28): the vision start token, the
placeholder tokens and the vision end token become one marker that keeps the
placeholder count, `<image:N>`.

The packer runs this file as a subprocess in the right venv:
    <venv python> promptcheck.py '<json request>'   →   {"canonical": "…"}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SENTINEL = "PIGEON7PROMPT7CHECK7ANSWER"
# The Qwen VL families write one picture this way in a rendered prompt.
_QWEN_PICTURE = re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>")


def snapshot(repo: str, hf_home: str | None) -> Path:
    """The local folder of a Hugging Face repo, without the network: a path as
    given, else HF_HOME's hub cache at refs/main."""
    p = Path(repo)
    if p.is_dir():
        return p
    hub = Path(hf_home or Path.home() / ".cache/huggingface") / "hub" / ("models--" + repo.replace("/", "--"))
    rev = (hub / "refs" / "main").read_text().strip()
    return hub / "snapshots" / rev


def revision(repo: str, hf_home: str | None) -> str | None:
    """The commit the local snapshot is at, or None for a plain folder."""
    if Path(repo).is_dir():
        return None
    try:
        return snapshot(repo, hf_home).name
    except OSError:
        return None


def precision(repo: str, hf_home: str | None) -> str | None:
    """`trained_from.precision` (§7.1) from the weights' own config.json:
    quantised bits win; otherwise the stored dtype."""
    try:
        cfg = json.loads((snapshot(repo, hf_home) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    q = cfg.get("quantization") or cfg.get("quantization_config")
    if isinstance(q, dict) and q.get("bits"):
        return f"{int(q['bits'])}bit"
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype")
                or (cfg.get("text_config") or {}).get("torch_dtype")
                or (cfg.get("text_config") or {}).get("dtype") or "").lower()
    return {"bfloat16": "bf16", "float16": "f16"}.get(dtype)


def text_canonical(repo: str, hf_home: str | None, user: str, system: str | None = None,
                   shots: list | None = None) -> str:
    """The user turn as mlx-lm rendered it into a training row, up to the
    answer — after the row's system message and fixed turns, when it had them
    (D85, D86)."""
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(snapshot(repo, hf_home))
    before = ([{"role": "system", "content": system}] if system is not None else []) + \
        [m for t in shots or [] for m in ({"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["pigeon"]})]
    full = tok.apply_chat_template(before + [{"role": "user", "content": user},
                                             {"role": "assistant", "content": SENTINEL}], tokenize=False)
    return full[:full.index(SENTINEL)]


def picture_canonical(repo: str, hf_home: str | None, image: str, text: str, limits: dict | None = None) -> str:
    """A picture row as mlx-vlm rendered and tokenised it in training, with
    the projector's picture size limits that training used (D80)."""
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    model, proc = load(str(snapshot(repo, hf_home)), lazy=True)
    if limits:
        _apply_limits(proc, limits)
    cfg = model.config.__dict__ if hasattr(model.config, "__dict__") else model.config
    conv = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": [{"type": "text", "text": SENTINEL}]}]
    full = apply_chat_template(proc, cfg, conv, add_generation_prompt=False, num_images=1)
    prompt = full[:full.index(SENTINEL)]
    token_id = cfg.get("image_token_id") or cfg.get("image_token_index")
    ids = prepare_inputs(processor=proc, images=[image], prompts=[prompt], image_token_index=token_id)["input_ids"]
    n = sum(1 for t in ids.reshape(-1).tolist() if int(t) == int(token_id))
    if len(_QWEN_PICTURE.findall(prompt)) != 1 or n < 1:
        raise ValueError(f"no canonical picture form for model type {cfg.get('model_type')!r} yet")
    return _QWEN_PICTURE.sub(f"<image:{n}>", prompt)


def _apply_limits(processor, lim: dict) -> None:
    """pigeon/picturesize.py apply_to_processor; repeated because this file
    runs on its own in the vision venv."""
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


def main(req: dict) -> dict:
    if req["mode"] == "picture":
        return {"canonical": picture_canonical(req["repo"], req.get("hf_home"), req["image"], req["text"],
                                               req.get("limits"))}
    return {"canonical": text_canonical(req["repo"], req.get("hf_home"), req["user"], req.get("system"), req.get("shots"))}


if __name__ == "__main__":
    try:
        print(json.dumps(main(json.loads(sys.argv[1]))))
    except Exception as e:  # the packer reports why and packs without prompt_check
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
