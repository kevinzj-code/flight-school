"""A vision run that dies late must not throw away the adapter it already made.

The case is measured, not imagined. The pilot's dog-breed run reached 0.0007
train loss by iter 300 and then died at iter 320 of 600 on a Metal
out-of-memory error. A person rescued the iter-300 checkpoint by hand and the
cartridge flew 36/36. Doing that by hand is not a workflow, and before this the
tool raised and lost the run.

This does not weaken "the test decides which checkpoint ships": the check ride
still runs on whatever gets promoted and is still the arbiter. What it prevents
is discarding a finished-enough adapter because the process exited non-zero
after producing it.

    python3 -m pytest src-tauri/python/bench/test_vision_salvage.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from instructor.train_vision import _newest_checkpoint  # noqa: E402


def _write(dir_: Path, name: str, body: str = "x") -> Path:
    p = dir_ / name
    p.write_text(body)
    return p


def test_the_highest_numbered_checkpoint_wins(tmp_path):
    """Not the newest by mtime — mlx-vlm can rewrite an earlier file, and the
    iteration number is the thing that means 'more trained'."""
    _write(tmp_path, "0000100_adapters.safetensors")
    _write(tmp_path, "0000300_adapters.safetensors")
    _write(tmp_path, "0000200_adapters.safetensors")
    step, path = _newest_checkpoint(tmp_path)
    assert step == 300
    assert path.name == "0000300_adapters.safetensors"


def test_the_pilots_own_layout_resolves_to_iter_300(tmp_path):
    """The exact directory the dog run left behind."""
    for n in ("0000100", "0000200", "0000300"):
        _write(tmp_path, f"{n}_adapters.safetensors")
    _write(tmp_path, "adapter_config.json", "{}")
    _write(tmp_path, "adapters.safetensors")
    assert _newest_checkpoint(tmp_path)[0] == 300


def test_no_checkpoints_means_no_salvage(tmp_path):
    """A run that died before writing anything has nothing to promote, and must
    keep raising rather than inventing an adapter."""
    _write(tmp_path, "adapter_config.json", "{}")
    assert _newest_checkpoint(tmp_path) is None


def test_the_final_weights_are_not_mistaken_for_a_checkpoint(tmp_path):
    """`adapters.safetensors` has no iteration number and is the file being
    replaced — reading it as a checkpoint would promote a file onto itself."""
    _write(tmp_path, "adapters.safetensors")
    assert _newest_checkpoint(tmp_path) is None


def test_unrelated_files_are_ignored(tmp_path):
    _write(tmp_path, "0000100_adapters.safetensors")
    _write(tmp_path, "notes.txt")
    _write(tmp_path, "0000100_adapters.safetensors.bak")
    step, path = _newest_checkpoint(tmp_path)
    assert (step, path.name) == (100, "0000100_adapters.safetensors")


def test_iteration_numbers_compare_numerically_not_as_text(tmp_path):
    """Zero padding makes string order agree with numeric order today. It must
    not be what the answer depends on."""
    _write(tmp_path, "90_adapters.safetensors")
    _write(tmp_path, "100_adapters.safetensors")
    assert _newest_checkpoint(tmp_path)[0] == 100
