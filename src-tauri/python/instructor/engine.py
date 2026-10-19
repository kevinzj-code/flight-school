"""llama-server lifecycle, the HTTP client, and the one-at-a-time rule.

Spec risk #2, in code: "9B generation + 4B training on one 16 GB machine only
works sequentially. If the app ever tries both at once it will thrash. Enforce
one-at-a-time in code."

So every resident model and every training run takes the same lock. The second
one does not queue and does not degrade — it refuses, and names what is holding
the lock. Flight School's training.rs does the same thing for the same reason.
"""

from __future__ import annotations

import json
import secrets
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import Config


class EngineBusy(RuntimeError):
    """Raised when something else already holds the machine."""


class EngineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# the exclusivity lock


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def engine_lock(cfg: Config, purpose: str) -> Iterator[Path]:
    """Hold the machine for `purpose`. One holder at a time, full stop."""
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    lock = cfg.runs_dir / ".engine.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held = {}
        try:
            held = json.loads(lock.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        pid = held.get("pid")
        if isinstance(pid, int) and _pid_alive(pid):
            raise EngineBusy(
                f"{held.get('purpose', 'another job')} is running (pid {pid}). "
                f"Training and generation do not fit on this machine at the same "
                f"time — wait for it, or stop it first."
            )
        # The holder died without cleaning up. Reclaim, but say so.
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, json.dumps(
            {"pid": os.getpid(), "purpose": purpose, "started": time.time()}
        ).encode())
        os.close(fd)
        yield lock
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# the server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Context window per server slot, and the headroom an extra slot must have
# before it is opened. Mirrors genserver.rs — the two paths serve the same
# models on the same machine and must not disagree about what fits.
GB_PER_EXTRA_SLOT = 0.75
MAX_SLOTS = 4

# Bigger Macs get a higher ceiling: (budget at or above, cap). 22 GB is a
# 32 GB machine's budget (32 * 0.75 - 1.5 = 22.5), 46 GB a 64 GB machine's.
# Below 32 GB nothing changes — 4 is what every run so far was measured at.
# The headroom arithmetic still has the last word; this only lifts the ceiling
# on machines whose headroom already reached it. Whether 8 or 16 sequences
# actually decode faster than 4 on a given chip is what timings.json is for.
SLOT_CAPS = ((46.0, 16), (22.0, 8))


def max_slots(budget_gb: float) -> int:
    for floor, cap in SLOT_CAPS:
        if budget_gb >= floor:
            return cap
    return MAX_SLOTS


def plan_slots(budget_gb: float, weights_gb: float) -> int:
    """How many requests this machine can afford in flight at once.

    Generation is bandwidth-bound on the weights, which are read once per
    decode step however many sequences share it, so a handful of concurrent
    requests costs roughly one request's time. What they cost is KV cache —
    the thing this app must never overcommit. Hence arithmetic against the
    real budget, and a fall back to 1 (today's behaviour exactly) whenever the
    headroom is not there. GB_PER_EXTRA_SLOT is a deliberate over-estimate:
    over-estimating costs a slot, under-estimating costs a thrashed run.
    """
    headroom = budget_gb - weights_gb - 1.0
    if headroom != headroom or headroom <= 0:      # NaN or nothing spare
        return 1
    return 1 + min(int(headroom // GB_PER_EXTRA_SLOT), max_slots(budget_gb) - 1)


def slots_for(cfg: Config, model: Path) -> int:
    """plan_slots against this config's generation budget and the model's own
    on-disk size — the only shape information available without parsing the
    GGUF, and a sound floor under what it occupies resident."""
    try:
        weights_gb = Path(model).stat().st_size / 1e9
    except OSError:
        return 1
    return plan_slots(cfg.generation_budget_gb, weights_gb)


@dataclass
class ServerSpec:
    model: Path
    # PER SLOT, not total: the server is started with ctx * n_parallel so that
    # opening slots never narrows the window any one request may use.
    ctx: int = 8192
    # One adapter, or several. Several is the two-character trick: all resident,
    # applied per request (ADAPTERS.md §4c). With init_without_apply the base
    # answers unmodified unless a request names an adapter — which makes adapter
    # support strictly additive, exactly as that doc recommends.
    lora: Path | list[Path] | None = None
    lora_init_without_apply: bool = False
    mmproj: Path | None = None
    n_gpu_layers: int = 999
    seed: int = 11
    # Qwen3.5 thinks before it answers, and llama-server counts those tokens
    # against the SAME budget as the answer. Measured on the 9B: 4 of 6 planning
    # calls spent the whole 6144-token budget reasoning and were cut off
    # mid-JSON — which reads as "the model cannot produce JSON" when it is
    # really "the budget was eaten before it started". Bounding the thinking
    # separately is the fix; -1 is unrestricted, 0 ends it immediately.
    reasoning_budget: int | None = None
    # Server slots. 1 is strictly one-at-a-time — the shape this app had before
    # slots existed, and still what a machine without memory headroom gets.
    n_parallel: int = 1


class LlamaServer:
    """A resident llama.cpp server, scoped to a `with` block.

    Every generation in this app goes through here: planning on the 9B,
    question drafting and answer distillation on the 4B, and the check ride on
    the 4B plus a freshly built adapter.
    """

    def __init__(self, cfg: Config, spec: ServerSpec, log: Path | None = None):
        self.cfg, self.spec, self.log_path = cfg, spec, log
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        # "stop" = the model finished; "length" = it ran out of budget. The
        # difference decides whether a missing plan is the model's fault or ours.
        self.last_finish_reason: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def api_key(self) -> str:
        if not getattr(self, "_api_key", None):
            self._api_key = secrets.token_hex(24)
        return self._api_key

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _args(self) -> list[str]:
        s = self.spec
        args = [
            str(self.cfg.llama_server),
            "-m", str(s.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            # A random key per server: another local process that finds the
            # port cannot use the model. /health stays open by the server's
            # own design, which is all the readiness wait needs.
            "--api-key", self.api_key,
            "-c", str(s.ctx * max(1, s.n_parallel)),
            "-ngl", str(s.n_gpu_layers),
            "--seed", str(s.seed),
            "--no-webui",          # nothing here needs a browser
            # Slots. Still ONE resident model — the exclusivity lock above is
            # what stops two models or a model and a training run sharing the
            # machine, and it is untouched. This only decides how much of the
            # GPU one resident model is allowed to use at a time.
            "-np", str(max(1, s.n_parallel)),
        ]
        if s.lora:
            loras = s.lora if isinstance(s.lora, list) else [s.lora]
            # ONE --lora with comma-separated paths. This build (b9851) warns
            # that a repeated flag keeps only the last value — and measured
            # behaviour with the repeated form was worse than the warning: both
            # adapters ended up APPLIED globally, so the "no adapter" state
            # answered as a two-personality mixture. Ids remain positional in
            # list order (robot=0, poet=1), same as the repeated form promised.
            for path in loras:
                if "," in str(path):
                    raise EngineError(f"adapter path contains a comma: {path}")
            args += ["--lora", ",".join(str(p) for p in loras)]
        if s.lora_init_without_apply:
            args += ["--lora-init-without-apply"]
        if s.mmproj:
            args += ["--mmproj", str(s.mmproj)]
        if s.reasoning_budget is not None:
            args += ["--reasoning-budget", str(s.reasoning_budget)]
        return args

    def start(self, timeout: float = 300.0) -> "LlamaServer":
        if not self.cfg.llama_server.exists():
            raise EngineError(f"llama-server not found at {self.cfg.llama_server}")
        if not self.spec.model.exists():
            raise EngineError(f"model not found: {self.spec.model}")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self._args(),
            stdout=self._log_fh or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if self._log_fh else subprocess.DEVNULL,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = ""
                if self.log_path and self.log_path.exists():
                    tail = "\n".join(self.log_path.read_text(errors="replace")
                                     .splitlines()[-12:])
                raise EngineError(
                    f"llama-server exited with code {self.proc.returncode}.\n{tail}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(0.5)
        self.stop()
        raise EngineError(f"llama-server did not become healthy within {timeout:.0f}s")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- requests ----------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float = 900.0) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            raise EngineError(f"{path} → {exc.code}: {exc.read()[:400].decode(errors='replace')}")

    # Qwen3.5 is a REASONING model. Left alone it emits several hundred tokens
    # of `reasoning_content` before a word of `content` — measured here at 3.0s
    # vs 0.2s for a one-word answer. Two consequences, both load-bearing:
    #
    #   1. A max_tokens budget sized for the ANSWER starves the answer entirely,
    #      because thinking spends the budget first. That is the silent-truncation
    #      failure wearing a new hat.
    #   2. Training data must be the answer WITHOUT the thinking, and the check
    #      ride must then run the same way — a cartridge that was taught to answer
    #      directly, evaluated under a template that invites it to ruminate, is
    #      being measured on a shape it never saw.
    #
    # So thinking is OFF by default everywhere, and turned on deliberately for
    # planning, where one slow call buys a better plan.
    @staticmethod
    def _think_kwargs(think: bool) -> dict[str, Any]:
        return {"chat_template_kwargs": {"enable_thinking": bool(think)}}

    def chat_full(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        think: bool = False,
        json_schema: dict | None = None,
        grammar: str | None = None,
        lora: list[dict] | None = None,
        timeout: float = 900.0,
    ) -> tuple[str, str]:
        """Returns (content, reasoning_content).

        json_schema turns on llama.cpp's constrained decoding: the sampler can
        only emit tokens that keep the output valid against the schema, so
        malformed or runaway JSON becomes structurally impossible. This is the
        contract enforced at generation time instead of after it.
        """
        payload: dict[str, Any] = {
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Reusing KV across requests that DIFFER in adapter state can carry
            # the other configuration's state across — measured, ADAPTERS.md.
            # The judge does exactly that (it re-asks with the adapter scaled to
            # zero), so caching is refused for any request naming an adapter
            # rather than left to each call site to remember. Flight School's
            # gen_chat applies the same rule for the same reason.
            "cache_prompt": lora is None,
            **self._think_kwargs(think),
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": json_schema},
            }
        if grammar is not None:
            # llama.cpp GBNF: the sampler may only emit text the grammar admits.
            # The check ride uses it to keep a label answer inside labels.txt
            # (spec §8.4, output_contract.constrained).
            payload["grammar"] = grammar
        if lora is not None:
            # Per-request adapter selection (ADAPTERS.md §4c, verified there):
            # overrides the load-time scales for this request only. Passing
            # [{"id": 0, "scale": 0.0}] gets the CLEAN BASE from a server that
            # has an adapter resident — no reload, no second model.
            payload["lora"] = lora
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = seed
        data = self._post("/v1/chat/completions", payload, timeout=timeout)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as exc:
            raise EngineError(f"unexpected completion shape: {json.dumps(data)[:300]}") from exc
        self.last_finish_reason = choice.get("finish_reason")
        return (msg.get("content") or "", msg.get("reasoning_content") or "")

    def chat(self, messages: Sequence[dict[str, Any]], **kw: Any) -> str:
        return self.chat_full(messages, **kw)[0]

    def tokenize(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text}, timeout=120).get("tokens", [])


@contextmanager
def serve(cfg: Config, spec: ServerSpec, purpose: str,
          log: Path | None = None) -> Iterator[LlamaServer]:
    """Take the lock, run the server, always give both back."""
    with engine_lock(cfg, purpose):
        srv = LlamaServer(cfg, spec, log=log)
        try:
            srv.start()
            yield srv
        finally:
            srv.stop()
