"""The `.pigeon` file — spec sheet §3, with §3.7's corrections built in.

A run used to end with a bare `adapter.lora.gguf` and a `report.json` beside
it. The base it fits, the exact prompt shape it was trained with and its
output contract lived in the run folder, and were lost the moment the file was
copied anywhere. That is the whole reason an adapter is hard to hand to
someone: an adapter is not portable; a `.pigeon` is.

    pigeon.json            required · the manifest
    adapter.lora.gguf      required
    labels.txt             required when kind = classifier
    system_prompt.txt      required when kind = persona
    icon.png               recommended
    README.md              recommended
    report.json            optional · the scorecard, control included
    examples/              optional · 3–5 inputs with expected outputs

Three modules, one contract:

    manifest   writes pigeon.json FROM THE RUN'S OWN CONFIG — the base's
               identity read out of its GGUF header, the sha256 of the file it
               was actually trained against, the prompt shape the check ride
               used. Nothing is retyped by hand.
    pack       zips it, store mode, files at the root, no wrapper folder.
    validate   the import check Pigeon will copy. Its test file is
               the cross-app contract. Encodes the two corrections: an exact
               hash is a FAST PATH, not the refusal rule — the same base at a
               different quantization warns and loads, a genuinely different
               base refuses and names the one needed; and a self-test claims a
               verdict only for label-shaped kinds.

Plain Python, standard library only, so the validator can be lifted into
another codebase whole.
"""

from .gguf import read_header, base_identity, quant_name
from .manifest import build_manifest, FORMAT_VERSION, KINDS
from .pack import pack_run, draft_readme, pick_examples, default_icon_png
from . import store, nest
from .validate import validate_pigeon, ValidationReport, BASE_EXACT, BASE_SAME_OTHER_QUANT, BASE_WRONG, BASE_MISSING

__all__ = [
    "read_header", "base_identity", "quant_name",
    "build_manifest", "FORMAT_VERSION", "KINDS",
    "pack_run", "draft_readme", "pick_examples", "default_icon_png",
    "validate_pigeon", "ValidationReport", "store", "nest",
    "BASE_EXACT", "BASE_SAME_OTHER_QUANT", "BASE_WRONG", "BASE_MISSING",
]
