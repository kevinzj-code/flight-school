"""D80: pictures the same size in training and in use.

    python3 -m pytest src-tauri/python/bench/test_picturesize.py -q
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pigeon import pack_run, picturesize  # noqa: E402

from bench.test_pigeon import QWEN35, base_q4, run_folder, write_gguf  # noqa: E402


def projector(tmp, **kv):
    return write_gguf(tmp / "mmproj.gguf", {"general.architecture": "clip", "clip.projector_type": "qwen3vl_merger",
                                            "clip.vision.patch_size": 16, "clip.vision.spatial_merge_size": 2, **kv},
                      kind="mmproj")


def test_limits_come_from_the_projector_not_the_processor(tmp_path):
    # Qwen3-VL-2B's projector names no limits: llama.cpp's rule for its type,
    # 8 to 4096 merged 32-pixel patches.
    assert picturesize.limits(projector(tmp_path)) == {"min_pixels": 8192, "max_pixels": 4194304, "unit": 32}
    # A projector that names them wins.
    named = projector(tmp_path, **{"clip.vision.image_min_pixels": 1024, "clip.vision.image_max_pixels": 65536})
    assert picturesize.limits(named) == {"min_pixels": 1024, "max_pixels": 65536, "unit": 32}
    # A qwen2.5vl merger: 14-pixel patches, so its unit is 28.
    q25 = write_gguf(tmp_path / "q25.gguf", {"general.architecture": "clip", "clip.projector_type": "qwen2.5vl_merger",
                                             "clip.vision.patch_size": 14}, kind="clip-vision")
    assert picturesize.limits(q25) == {"min_pixels": 8 * 28 * 28, "max_pixels": 4096 * 28 * 28, "unit": 28}
    with pytest.raises(ValueError, match="no picture size rule"):
        picturesize.limits(write_gguf(tmp_path / "x.gguf", {"clip.projector_type": "llava"}, kind="mmproj"))


def test_the_limits_replace_the_processors_own():
    """Qwen3-VL's processor came with size.shortest_edge 65536: every picture
    enlarged to 256 x 256. The projector's limits replace it, in both spellings."""
    class ImageProcessor:
        def __init__(self):
            self.size = {"shortest_edge": 65536, "longest_edge": 16777216}
            self.min_pixels, self.max_pixels = 65536, 16777216

    class Processor:
        image_processor = ImageProcessor()
    p = Processor()
    picturesize.apply_to_processor(p, {"min_pixels": 8192, "max_pixels": 4194304, "unit": 32})
    ip = p.image_processor
    assert ip.size == {"shortest_edge": 8192, "longest_edge": 4194304} and (ip.min_pixels, ip.max_pixels) == (8192, 4194304)


def _picture_pack(tmp_path, served):
    base = base_q4(tmp_path, QWEN35, "Qwen3.5-4B-Q4_K_M.gguf")
    rd = run_folder(tmp_path, base, vision=True)
    mm = projector(tmp_path)
    (rd / "config.toml").write_text(f'[models]\ntarget_mmproj = "{mm}"\n')
    canonical = "<|im_start|>user\n<image:64>What do you do next?<|im_end|>\n<|im_start|>assistant\n"

    def builder(run_dir, manifest, facts, example):
        return {"input": {"text": "What do you do next?", "image": "examples/001.jpg"}, "canonical": canonical}, None
    out = tmp_path / "p.pigeon"
    return out, lambda: pack_run(rd, out, prompt_check_builder=builder, picture_counter=lambda *a: served)


def test_a_count_mismatch_refuses_the_pack(tmp_path):
    out, pack = _picture_pack(tmp_path, served=25)
    with pytest.raises(ValueError) as e:
        pack()
    assert "64 tokens in training but 25 in llama.cpp" in str(e.value)
    assert not out.exists() and not out.with_suffix(".pigeon.part").exists()


def test_a_matching_count_packs(tmp_path):
    out, pack = _picture_pack(tmp_path, served=64)
    pack()
    with zipfile.ZipFile(out) as z:
        assert "<image:64>" in json.loads(z.read("pigeon.json"))["prompt_check"]["canonical"]


MODELS = Path.home() / "Library/Application Support/mode.airoplane.pigeon/models"


@pytest.mark.skipif(not (MODELS / "Qwen3VL-2B-Instruct-Q4_K_M.gguf").is_file(), reason="Qwen3-VL-2B is not installed")
def test_llama_cpp_counts_a_160_pixel_board_as_25(tmp_path):
    """The real count the packer relies on, on this Mac's own base."""
    from instructor.config import load
    from instructor.engine import LlamaServer, ServerSpec
    runs = Path.home() / "Library/Application Support/mode.airoplane.flightschool/bakeoff/runs/seed-run-v2-qwen3vl-app"
    board = Path.home() / "Library/Application Support/mode.airoplane.flightschool/quickbuild/runs/qb-1790288577184/recording/s1/f000002.jpg"
    if not (runs / "config.toml").is_file() or not board.is_file():
        pytest.skip("the Seed Run V2 run is not on this Mac")
    cfg = load(runs / "config.toml")
    with LlamaServer(cfg, ServerSpec(model=cfg.target_gguf, mmproj=cfg.target_mmproj, ctx=4096), log=tmp_path / "s.log") as srv:
        assert picturesize.count_in_llama(srv, board, "What do you do next?") == 25
