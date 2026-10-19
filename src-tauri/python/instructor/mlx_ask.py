#!/usr/bin/env python3
"""Ask a trained adapter its exam in MLX — the runtime it trained in.

Run by instructor/mlx_ride.py in the venv that trained the run (mlx-lm for
text, mlx-vlm for pictures), never imported by the app's own process:

    <venv python> mlx_ask.py '<json request>'   →   {"answers": [...]}

Each prompt is the training row's own prefix: the row rendered by the trainer's
chat-template path with a sentinel answer, cut where the answer starts — the
same construction as pigeon/promptcheck.py, so the prompt is exactly what
training saw (a template's empty reasoning block included). Greedy decoding,
the ride's token budget, no grammar: MLX has none to match llama.cpp's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SENTINEL = "PIGEON7PROMPT7CHECK7ANSWER"


def snapshot(repo: str, hf_home: str | None) -> Path:
    p = Path(repo)
    if p.is_dir():
        return p
    hub = Path(hf_home or Path.home() / ".cache/huggingface") / "hub" / ("models--" + repo.replace("/", "--"))
    return hub / "snapshots" / (hub / "refs" / "main").read_text().strip()


def ask_text(req: dict) -> list[str]:
    from mlx_lm import generate, load
    model, tok = load(str(snapshot(req["repo"], req.get("hf_home"))), adapter_path=req["adapter_dir"])
    out = []
    for row in req["rows"]:
        full = tok.apply_chat_template([{"role": "user", "content": row["text"]},
                                        {"role": "assistant", "content": SENTINEL}], tokenize=False)
        prompt = full[:full.index(SENTINEL)]
        out.append(generate(model, tok, prompt, max_tokens=req["max_tokens"], verbose=False))
    return out


def ask_pictures(req: dict) -> list[str]:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc = load(str(snapshot(req["repo"], req.get("hf_home"))), adapter_path=req["adapter_dir"])
    if req.get("limits"):
        # D80: the projector's picture size limits, as training used them.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pigeon.picturesize import apply_to_processor
        apply_to_processor(proc, req["limits"])
    cfg = model.config.__dict__ if hasattr(model.config, "__dict__") else model.config
    out = []
    for row in req["rows"]:
        conv = [{"role": "user", "content": [{"type": "image", "image": row["image"]}, {"type": "text", "text": row["text"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": SENTINEL}]}]
        full = apply_chat_template(proc, cfg, conv, add_generation_prompt=False, num_images=1)
        g = generate(model, proc, full[:full.index(SENTINEL)], image=[row["image"]],
                     max_tokens=req["max_tokens"], temperature=0.0, verbose=False)
        out.append(g.text if hasattr(g, "text") else str(g))
    return out


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        answers = ask_pictures(request) if request["mode"] == "picture" else ask_text(request)
        print(json.dumps({"answers": answers}))
    except Exception as e:  # the ride reports why and keeps its llama.cpp result
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
