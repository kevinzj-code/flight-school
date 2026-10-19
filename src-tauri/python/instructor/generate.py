"""One training row: a question, its answer, its group, and for a picture job
the image it is about (a path inside the corpus)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Row:
    question: str
    answer: str
    group: str
    image: str | None = None      # relative path inside the corpus, vision only

    def to_json(self) -> dict[str, Any]:
        d = {"question": self.question, "answer": self.answer, "group": self.group}
        if self.image:
            d["image"] = self.image
        return d
