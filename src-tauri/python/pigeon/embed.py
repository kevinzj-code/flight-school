"""`hybrid-v1`: a Reader's material embedded once, at pack time, so the app
that answers only ever embeds the question.

The embedding model is a second pinned identity beside the base
(docs/PIGEON-FORMAT.md §8.6): its file name and sha256 travel in the manifest,
and a reader that has a different model, or none, refuses the method by name
rather than searching with something else. One model, one set of prefixes,
one pooling — the vectors in the file were made this way and the question
must be too.

Measured 2026-09-30 on the Constitution Reader's 176 questions, right section
first: keyword-overlap-v2 117, this model on whole sections 117, on paragraphs
126, and the blend below 144 (top three: 164). Indexing 52 sections took 1.2 s;
a question embeds in 8 ms.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import socket
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = {
    "name": "nomic-embed-text-v1.5",
    "file": "nomic-embed-text-v1.5.Q8_0.gguf",
    "sha256": "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7",
    "dims": 768,
    "pooling": "mean",
    "document_prefix": "search_document: ",
    "query_prefix": "search_query: ",
}
RETRIEVAL = "hybrid-v1"
# The blend, and the floor under which the honest answer is no section:
# measured on the right section's cosine (p5 0.658, min 0.537) against
# off-topic questions (0.49–0.61). One definition, mirrored in chunker.rs and
# Pigeon's material.rs.
COSINE_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3
COSINE_FLOOR = 0.58
EMBEDDINGS_FILE = "material/embeddings.json"


def model_path(model_dirs) -> Path | None:
    """The embedding model, if the importing app has it — by name, the hash
    checked once here so a wrong file of the right name is not used."""
    for d in model_dirs or []:
        p = Path(d) / MODEL["file"]
        if p.is_file() and sha256_of(p) == MODEL["sha256"]:
            return p
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paragraphs(text: str) -> list[str]:
    """The units a section is embedded as: its paragraphs. Pigeon scores a
    section by its best paragraph, which is how "how old must the president
    be" reaches the one clause about the Age of thirty five Years inside a
    663-word section."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def pack_f16(vec) -> str:
    return base64.b64encode(struct.pack(f"<{len(vec)}e", *vec)).decode("ascii")


def unpack_f16(s: str, dims: int) -> list[float]:
    raw = base64.b64decode(s)
    if len(raw) != dims * 2:
        raise ValueError(f"a unit holds {len(raw) // 2} numbers, not {dims}")
    return list(struct.unpack(f"<{dims}e", raw))


def normalise(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def chunks_digest(index: dict) -> str:
    """What the units were made from, so a stale embeddings.json is noticed:
    the ordinals, titles and texts, nothing else."""
    h = hashlib.sha256()
    for c in index.get("chunks", []):
        h.update(json.dumps([c.get("ordinal"), c.get("title", ""), c.get("text", "")], ensure_ascii=False).encode())
    return h.hexdigest()


class EmbeddingServer:
    """llama-server in embedding mode, for the time it takes to index."""

    def __init__(self, llama_server: Path | str, model: Path | str, *, ctx: int = 2048, log: Path | None = None):
        self.bin, self.model, self.ctx, self.log = Path(llama_server), Path(model), ctx, log
        self.port = self._free_port()
        self.proc: subprocess.Popen | None = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def __enter__(self):
        args = [str(self.bin), "-m", str(self.model), "--embedding", "--pooling", MODEL["pooling"],
                "-c", str(self.ctx), "-b", str(self.ctx), "-ub", str(self.ctx),
                "--port", str(self.port), "--host", "127.0.0.1", "-ngl", "999"]
        log = open(self.log, "ab") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(args, stdout=log, stderr=log)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("the embedding server stopped before it was ready")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if b'"ok"' in r.read():
                        return self
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("the embedding server did not become ready in two minutes")

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def embed(self, text: str) -> list[float]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/embeddings",
                                     data=json.dumps({"input": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["data"][0]["embedding"]


def index_material(index: dict, srv: EmbeddingServer) -> dict:
    """`material/embeddings.json` for a chunk index: one unit per paragraph,
    the section's title in front of each so a paragraph knows where it is."""
    units = []
    for c in index["chunks"]:
        for part, para in enumerate(paragraphs(c["text"])):
            vec = normalise(srv.embed(f"{MODEL['document_prefix']}{c.get('title', '')}\n{para}"))
            units.append({"ordinal": int(c["ordinal"]), "part": part, "f16": pack_f16(vec)})
    return {"model": dict(MODEL), "chunks_sha256": chunks_digest(index), "units": units}


def ensure_embeddings(material_dir: Path | str, *, llama_server, model_dirs, log: Path | None = None) -> tuple[Path | None, str | None]:
    """`<material>/embeddings.json`, made or refreshed when the chunks changed.
    Returns (path, None) when it is there, or (None, why) when it cannot be —
    no server, no model — so the packer can say so and pack the lexical
    method instead."""
    material_dir = Path(material_dir)
    chunks_path = material_dir / "chunks.json"
    if not chunks_path.is_file():
        return None, "no material"
    index = json.loads(chunks_path.read_text())
    out = material_dir / "embeddings.json"
    digest = chunks_digest(index)
    if out.is_file():
        try:
            have = json.loads(out.read_text())
            if have.get("chunks_sha256") == digest and (have.get("model") or {}).get("sha256") == MODEL["sha256"]:
                return out, None
        except ValueError:
            pass
    if not llama_server or not Path(llama_server).is_file():
        return None, "no llama-server to embed with"
    model = model_path(model_dirs)
    if model is None:
        return None, f"the embedding model {MODEL['file']} is not on this Mac — download it in Pigeon, then send again"
    with EmbeddingServer(llama_server, model, log=log) as srv:
        data = index_material(index, srv)
    out.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return out, None


def embeddings_problems(raw: bytes, chunk_ordinals, *, model_sha: str | None = MODEL["sha256"]) -> list[str]:
    """`material/embeddings.json` checked the way Pigeon parses it. `model_sha`
    None leaves the which-model question to the caller (the import check names
    it `embedding_model_mismatch`, spec §9.4)."""
    try:
        e = json.loads(raw)
    except ValueError as err:
        return [f"{EMBEDDINGS_FILE} could not be read — {err}"]
    if not isinstance(e, dict):
        return [f"{EMBEDDINGS_FILE} must be an object"]
    m = e.get("model")
    if not isinstance(m, dict) or not isinstance(m.get("file"), str) or not isinstance(m.get("sha256"), str) \
            or not re.fullmatch(r"[0-9a-f]{64}", m.get("sha256", "")) or not isinstance(m.get("dims"), int):
        return [f"{EMBEDDINGS_FILE} must name its model: file, sha256 (64 hex) and dims"]
    if model_sha is not None and m["sha256"] != model_sha:
        return [f"{EMBEDDINGS_FILE} was made with a model this version does not carry ({m['file']}) — update the app"]
    dims = m["dims"]
    units = e.get("units")
    if not isinstance(units, list) or not units:
        return [f"{EMBEDDINGS_FILE} has no units in it"]
    ords = set(chunk_ordinals)
    for i, u in enumerate(units):
        if not isinstance(u, dict) or not isinstance(u.get("ordinal"), int) or isinstance(u.get("ordinal"), bool):
            return [f"{EMBEDDINGS_FILE} unit {i} needs a whole-number 'ordinal'"]
        if u["ordinal"] not in ords:
            return [f"{EMBEDDINGS_FILE} unit {i} belongs to section {u['ordinal']}, which chunks.json does not have"]
        try:
            unpack_f16(u.get("f16", ""), dims)
        except Exception as err:
            return [f"{EMBEDDINGS_FILE} unit {i}: {err}"]
    return []


# ── the search itself, for measuring and for tests; the apps carry ports ──

def zscores(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
    return [(x - m) / sd if sd > 0 else 0.0 for x in xs]


def hybrid_v1(chunks, units, qvec, lexical):
    """The best section's index, or None. `units` are (chunk index, vector);
    `lexical` the keyword-overlap-v2 score per chunk, in chunk order."""
    n = len(chunks)
    cos = [-1.0] * n
    for i, vec in units:
        cos[i] = max(cos[i], sum(a * b for a, b in zip(qvec, vec)))
    zc, zl = zscores(cos), zscores(lexical)
    best = max(range(n), key=lambda i: (COSINE_WEIGHT * zc[i] + LEXICAL_WEIGHT * zl[i], -i))
    return best if cos[best] >= COSINE_FLOOR else None
