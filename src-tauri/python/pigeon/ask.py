"""Using a Pigeon, rather than building one.

Flight School could package a `.pigeon` and never ask one a question — the whole
pipeline ended at the file. This is the other half: open a packed Pigeon, check
it the way any app must, put the base and the adapter in front of llama.cpp, and
return the answer already turned into whatever the Pigeon actually produces.

Three rules it enforces, all of them from docs/PIGEON-FORMAT.md rather than
invented here:

  * **The import check runs first.** A file this machine cannot load is refused
    before a server is started, with the message the format doc specifies.
  * **The template is reproduced byte for byte.** `prompt_template` is the shape
    the adapter was trained in; asking in another shape measures the prompt
    instead of the adapter (§2), and for a long exact answer it produces
    well-formed nonsense (the baseline notes, "The logit test").
  * **A skill renders by concatenation.** Nothing in the file is executed —
    `skill/skill.json` supplies strings and this module joins them (§9.2).

Standard library only, like the rest of this package, so it can be lifted whole
into Pigeon.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .gguf import same_base
from .validate import BASE_EXACT, BASE_SAME_OTHER_QUANT, installed_bases, validate_pigeon


class AskError(RuntimeError):
    """Raised with the message a person should see, never a stack trace."""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def render_with_skill(payload: dict[str, Any], recipe: dict[str, Any]) -> str:
    """A recipe plus the file's own skill payload, into the artefact.

    Concatenation, deliberately. See §9.2: a `.pigeon` arrives from someone else
    and an app that executed code out of one would have discarded the
    validate-then-load doctrine the format rests on.
    """
    parts = payload.get("parts") or {}
    frags = [payload.get("head", "")]
    for key in payload.get("order") or list(parts):
        value = recipe.get(key)
        if not value:
            continue
        table = parts.get(key) or {}
        names = value if isinstance(value, list) else [value]
        for name in names:
            if name not in table:
                # An unknown part is a miss, never repaired to the nearest known
                # one — see §9.3. A renderer that repairs cannot be trusted to
                # have drawn what the model asked for.
                raise AskError(f"the answer names {key}={name!r}, which this "
                               f"Pigeon's skill does not carry")
            frags.append(table[name])
    return str(payload.get("wrapper", "{parts}")).replace("{parts}", "".join(frags))


@contextmanager
def _server(llama_server: Path, base: Path, adapter_bytes: bytes,
            ctx: int, tmp: Path, ready_within: float = 180) -> Iterator[str]:
    adapter = tmp / "adapter.lora.gguf"
    adapter.write_bytes(adapter_bytes)
    port = _free_port()
    key = secrets.token_hex(24)
    proc = subprocess.Popen(
        [str(llama_server), "--model", str(base), "--lora", str(adapter),
         "--port", str(port), "--ctx-size", str(ctx), "-ngl", "99",
         "--host", "127.0.0.1", "--no-webui", "--api-key", key],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    try:
        deadline = time.time() + ready_within
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AskError("the engine stopped before it was ready")
            try:
                urllib.request.urlopen(f"{url}/health", timeout=2).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            raise AskError(f"the engine did not come up within {ready_within:.0f} seconds")
        yield url, headers
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def ask(pigeon: Path | str, text: str, *, llama_server: Path | str,
        models: list[Path] | None = None, ctx: int = 2048,
        max_tokens: int | None = None, temperature: float = 0.0,
        timeout_s: float | None = None) -> dict[str, Any]:
    """Ask one packed Pigeon one question. Returns what it answered and, when it
    carries a skill, what that answer renders to.

    `timeout_s` bounds the whole ask — engine start and answer — and is kept
    HERE rather than by whoever calls: killing this process from outside
    would leave its llama-server running, while running out of time in here
    still goes through the `finally` that stops it."""
    started = time.time()
    pigeon = Path(pigeon)
    report = validate_pigeon(pigeon, installed_bases(models or []))
    if not report.ok:
        raise AskError("; ".join(report.errors))
    if report.base_match not in (BASE_EXACT, BASE_SAME_OTHER_QUANT):
        raise AskError(f"the base this Pigeon needs is {report.base_match} on "
                       f"this machine: {report.base_needed}")
    manifest = report.manifest or {}
    # The import check reports WHICH KIND of match it found; finding the file
    # again is the app's job, by the same rule §4 gives — the exact file first,
    # then any GGUF of the same base identity.
    wanted = manifest.get("base_model") or {}
    bases = installed_bases(models or [])
    base = next((Path(b["path"]) for b in bases if b["file"] == wanted.get("file")), None)
    if base is None:
        base = next((Path(b["path"]) for b in bases if same_base(b, wanted)), None)
    if base is None or not base.exists():
        raise AskError(f"this Pigeon needs {wanted.get('file')} and this machine "
                       f"does not have it")

    template = str(manifest.get("prompt_template", "{input}"))
    if "{material}" in template:
        # Grounded Pigeons need the app to retrieve and paste first; that is a
        # different call and this one must not pretend to make it.
        raise AskError("this Pigeon answers from material it is handed; ask it "
                       "through the grounded path, not this one")
    prompt = template.replace("{input}", text)
    budget = max_tokens or int((manifest.get("output_contract") or {}).get("max_tokens") or 64)

    with zipfile.ZipFile(pigeon) as z:
        adapter_bytes = z.read("adapter.lora.gguf")
        skill = json.loads(z.read("skill/skill.json")) if manifest.get("skill") else None

    tmp = pigeon.parent / f".{pigeon.stem}.serving"
    tmp.mkdir(exist_ok=True)
    try:
        ready_within = min(180.0, timeout_s) if timeout_s else 180.0
        with _server(Path(llama_server), base, adapter_bytes, ctx, tmp, ready_within) as (url, headers):
            body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                               "temperature": temperature,
                               "max_tokens": budget}).encode()
            req = urllib.request.Request(f"{url}/v1/chat/completions", body, headers)
            left = max(5.0, timeout_s - (time.time() - started)) if timeout_s else 300.0
            reply = json.loads(urllib.request.urlopen(req, timeout=left).read())
        answer = reply["choices"][0]["message"]["content"].strip()
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()

    out: dict[str, Any] = {"answer": answer, "base": base.name,
                           "base_match": report.base_match,
                           "warnings": report.warnings}
    if skill:
        try:
            recipe = json.loads(answer)
        except json.JSONDecodeError:
            raise AskError(f"this Pigeon answers with a recipe and that was not "
                           f"JSON: {answer[:120]}")
        out["recipe"] = recipe
        out["rendered"] = render_with_skill(skill, recipe)
        out["skill"] = f"{skill.get('name')} v{skill.get('version')}"
    return out
