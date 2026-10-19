"""The build's shared pieces: the hooks a stage reports progress through, and
the image-token headroom a vision run adds to its sequence ceiling.

The stages themselves live in `quickbuild.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# Headroom for image tokens on a vision run. The text tokenizer cannot see them,
# and a ceiling that ignores them truncates every example at the picture.
VISION_IMAGE_TOKEN_ALLOWANCE = 1280


@dataclass
class Hooks:
    """Optional rendering callbacks. The build runs fine with none of them."""
    phase: Callable[[str, str], None] | None = None          # (title, detail)
    note: Callable[[str], None] | None = None
    warn: Callable[[str], None] | None = None
    question_progress: Callable[[str, int, int], None] | None = None
    answer_progress: Callable[[int, int], None] | None = None
    train_progress: Callable[[int, int, str, float | None], None] | None = None
    answer_sample: Callable[[str, str, str], None] | None = None  # (group, question, answer)

    def _say(self, fn, *a):
        if fn:
            fn(*a)
