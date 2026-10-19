"""`prompt_check`, the Runner's half — spec §8.6.

`promptcheck.py` WRITES `canonical`, with the trainer's own code path. This
module is the other side: it rebuilds the prompt as a Runner does (§8.1–§8.3,
`runner.messages`), renders it through the base's own chat template, writes it
in canonical form and compares it with `canonical` byte for byte. A Pigeon whose
prompt differs is not run (`prompt_mismatch`).

Rendering needs `jinja2`, set up as transformers and llama.cpp both run a chat
template (`trim_blocks`, `lstrip_blocks`); it is imported only when a template
is rendered, so the rest of the package stays standard library only.
"""
from __future__ import annotations

from . import runner


def render_chat(template: str, messages: list[dict]) -> str:
    """`messages` through a GGUF's `tokenizer.chat_template`, assistant turn opened."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=True)


def canonical(manifest: dict, text: str, chat_template: str, *, system_prompt: str | None = None,
              sections: list[dict] | None = None, history: list[dict] | None = None,
              picture: dict | None = None, shots: list[dict] | None = None) -> str:
    """The prompt for one input in canonical form. `picture` is
    `{"markup", "tokens"}`: how this base's template writes one picture, and the
    number of placeholder tokens this picture becomes; each is written `<image:N>`."""
    msgs = []
    for m in runner.messages(manifest, text, system_prompt=system_prompt, sections=sections, history=history,
                             shots=shots):
        content = m["content"]
        if m.get("image_first"):
            content = [{"type": "image"}, {"type": "text", "text": content}]
        msgs.append({"role": m["role"], "content": content})
    out = render_chat(chat_template, msgs)
    if picture:
        out = out.replace(picture["markup"], f"<image:{int(picture['tokens'])}>")
    return out


def check(manifest: dict, chat_template: str, *, system_prompt: str | None = None,
          chunks: list[dict] | None = None, picture: dict | None = None, shots: list[dict] | None = None) -> dict:
    """{"ok", "warnings", "rendered", "position"}: whether the prompt this Runner
    builds for `prompt_check.input` is the one training built. Always without
    history. `position` is the first differing byte of the UTF-8 text, counted
    from 0 (D53), or None."""
    pc = manifest.get("prompt_check")
    if pc is None:
        return {"ok": True, "warnings": ["prompt_unchecked"], "rendered": None, "position": None}
    given, sections = pc["input"], None
    if isinstance(given, dict) and "image" in given:
        text = given["text"]
    elif isinstance(given, dict):
        by_ordinal = {c["ordinal"]: c for c in chunks or []}
        text, sections = given["input"], [by_ordinal[o] for o in given.get("sections", [])]
    else:
        text = given
    built = canonical(manifest, text, chat_template, system_prompt=system_prompt, sections=sections, picture=picture,
                      shots=shots)
    want = pc["canonical"]
    if built == want:
        return {"ok": True, "warnings": [], "rendered": built, "position": None}
    mine, theirs = built.encode("utf-8"), want.encode("utf-8")
    position = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
    return {"ok": False, "warnings": [], "rendered": built, "position": position}
