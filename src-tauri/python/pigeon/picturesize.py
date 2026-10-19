"""Pictures the same size in training and in use — spec D80 (§8.6, §13.1).

A picture Pigeon is trained in MLX and served in llama.cpp. Each resizes a
picture between a smallest and a largest size before the model sees it, and
the two must agree, or the model is trained on pictures it never sees. Found
by the engine: MLX's Qwen3-VL processor enlarged a 160-pixel board to 256
(64 tokens); llama.cpp did not (25 tokens).

So the limits come from the base's own projector file, never from the
training processor's defaults (`limits`), and before a Pigeon is packed the
picture is counted in llama.cpp itself (`count_in_llama`).
"""
from __future__ import annotations

import base64
from pathlib import Path

from .gguf import read_header

#: Qwen-VL projectors: llama.cpp's own limit when the header names none —
#: from 8 to 4096 merged patches, each (patch_size × spatial_merge_size)²
#: pixels. For Qwen3-VL-2B: 8 × 32² = 8192 and 4096 × 32² = 4,194,304.
_QWEN_TOKENS = (8, 4096)
_QWEN_TYPES = ("qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger")


def limits(mmproj: Path | str) -> dict:
    """{"min_pixels", "max_pixels", "unit"} for a projector GGUF: the header's
    `clip.vision.image_min_pixels` / `image_max_pixels` when it has them, else
    llama.cpp's rule for its projector type. `unit` is the side, in pixels, of
    one merged patch (one token): sizes are rounded to it."""
    h = read_header(mmproj)
    patch = int(h.get("clip.vision.patch_size") or 0)
    merge = int(h.get("clip.vision.spatial_merge_size") or 2)
    kind = str(h.get("clip.projector_type") or "")
    if not patch or kind not in _QWEN_TYPES:
        raise ValueError(f"{Path(mmproj).name}: no picture size rule for projector type {kind!r}")
    unit = patch * merge
    lo, hi = h.get("clip.vision.image_min_pixels"), h.get("clip.vision.image_max_pixels")
    if not (isinstance(lo, int) and isinstance(hi, int)):
        lo, hi = _QWEN_TOKENS[0] * unit * unit, _QWEN_TOKENS[1] * unit * unit
    return {"min_pixels": int(lo), "max_pixels": int(hi), "unit": unit}


def apply_to_processor(processor, lim: dict) -> None:
    """Set an mlx-vlm / transformers Qwen-VL processor to the projector's
    limits. Both spellings are set, because processor versions read one or
    the other (Qwen3-VL's `size.shortest_edge` is a pixel count, not an edge)."""
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


def _data_uri(p: Path) -> str:
    kind = "png" if p.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(p.read_bytes()).decode()


def count_in_llama(srv, image: Path | str, text: str) -> int:
    """The placeholder tokens a picture becomes in llama.cpp: the prompt's
    token count with the picture, less the same prompt without it, less the
    vision start and end tokens, which llama.cpp counts with the picture
    (spec §8.6). `srv` is a running instructor.engine.LlamaServer with the
    base and its projector."""
    def prompt_tokens(content) -> int:
        reply = srv._post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}})
        return int(reply["usage"]["prompt_tokens"])
    with_picture = prompt_tokens([{"type": "image_url", "image_url": {"url": _data_uri(Path(image))}},
                                  {"type": "text", "text": text}])
    without = prompt_tokens([{"type": "text", "text": text}])
    return with_picture - without - 2


def mismatch(trained: int, served: int) -> str | None:
    """The refusal sentence when training and serving disagree, or None."""
    if trained == served:
        return None
    return (f"its check picture became {trained} tokens in training but {served} in llama.cpp, where it is served: "
            f"it would be trained on pictures it never sees in use (D80). Train it again with the projector's size limits")
