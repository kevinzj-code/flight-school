"""The window a run trains at, and whether it checkpoints, decided on that window.

Run: `python3 -m pytest src-tauri/python/bench/test_preflight.py -q`. No model,
no tokenizer: the shapes are the real bases' config.json figures, and the
budgets are what paths.rs gives a 16, 32 and 64 GB Mac (RAM * 0.75 - 1.5).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instructor.builder import VISION_IMAGE_TOKEN_ALLOWANCE  # noqa: E402
from instructor.preflight import ModelDims, shape_for  # noqa: E402

VISION_LARGE = ModelDims(hidden_size=2048, num_layers=36, weights_gb=3.07)   # a 3B-class picture model, 4-bit
TEXT_15B = ModelDims(hidden_size=1536, num_layers=28, weights_gb=3.09)    # Qwen2.5-1.5B-Instruct-bf16
HYBRID_4B = ModelDims(hidden_size=2560, num_layers=32, weights_gb=2.5, is_hybrid=True)  # Qwen3.5-4B
MAC_16, MAC_32, MAC_64 = 10.5, 22.5, 46.5


def test_a_picture_job_is_judged_at_the_window_it_trains_at():
    """The bug this pins: a picture job's window was widened for the image
    AFTER the memory verdict, so a 40-token question was judged at 128 tokens —
    "fits, no checkpointing" — and trained at 1664, where the same estimate
    says it does not fit without checkpointing on a 16 GB Mac."""
    max_seq, _, _, verdict, ckpt = shape_for(VISION_LARGE, 40, VISION_IMAGE_TOKEN_ALLOWANCE, MAC_16)
    assert max_seq == 1664, "the image allowance must be inside the window"
    assert ckpt, "a 16 GB Mac must checkpoint a picture window this long"
    assert verdict == "fits", "with checkpointing it fits; the run is not refused"


def test_a_picture_job_drops_checkpointing_where_there_is_room():
    for budget in (MAC_32, MAC_64):
        *_, ckpt = shape_for(VISION_LARGE, 40, VISION_IMAGE_TOKEN_ALLOWANCE, budget)
        assert not ckpt, f"a {budget} GB budget has room for this window without checkpointing"


def test_a_long_text_window_checkpoints_only_where_it_must():
    # About 2,000-token rows, the long text job MEASURED in engine.js.
    max_seq, *_, ckpt_16 = shape_for(TEXT_15B, 2000, 0, MAC_16)
    *_, ckpt_32 = shape_for(TEXT_15B, 2000, 0, MAC_32)
    assert max_seq == 2560
    assert ckpt_16 and not ckpt_32


def test_short_text_is_unchanged():
    max_seq, *_, ckpt = shape_for(TEXT_15B, 200, 0, MAC_16)
    assert max_seq == 256 and not ckpt


def test_a_hybrid_base_keeps_checkpointing_past_1024_whatever_the_room():
    """Measured on Qwen3.5: checkpointing made long windows faster, not just
    smaller. A 64 GB Mac's room is no reason to turn it off there."""
    max_seq, *_, ckpt = shape_for(HYBRID_4B, 1000, 0, 512.0)
    assert max_seq == 1280 and ckpt
    max_seq, *_, ckpt = shape_for(HYBRID_4B, 100, 0, MAC_64)
    assert max_seq == 128 and not ckpt, "a short window on a hybrid base still drops it when it fits"


def test_an_unknown_shape_checkpoints():
    """No config.json means no estimate; the safe direction is on."""
    *_, verdict, ckpt = shape_for(None, 200, 0, MAC_64)
    assert verdict == "unknown" and ckpt
