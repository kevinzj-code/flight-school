"""The `.nest` — a whole flock in one file.

A ZIP, store mode, of `.pigeon` files at the root, plus `nest.json` saying
what is inside. That is all it is on purpose: Pigeon may ignore
`nest.json` entirely and take the Pigeons, and a person with no app can
unzip it and see files they recognize.

    my-flock.nest
    ├── nest.json
    ├── coin-sorter.pigeon
    └── note-to-title.pigeon

Each Pigeon inside is validated on its own when the nest is added; a nest
with one bad member still yields the good ones, and says which was bad.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from datetime import date
from pathlib import Path

NEST_FORMAT_VERSION = 1


class NotANest(ValueError):
    """`code` is the spec's Appendix A code: `not_a_zip`, `bad_zip` (the nest
    breaks the strict ZIP rules, §3.2) or `not_a_nest` (everything else)."""

    def __init__(self, msg: str, code: str = "not_a_nest"):
        super().__init__(msg)
        self.code = code


def is_nest(path: Path | str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return "nest.json" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def _describe(pigeon: Path) -> dict:
    """What nest.json says about one member — read from the Pigeon's own
    manifest, never typed."""
    entry = {"file": pigeon.name, "bytes": pigeon.stat().st_size,
             "sha256": hashlib.sha256(pigeon.read_bytes()).hexdigest()}
    try:
        with zipfile.ZipFile(pigeon) as z:
            m = json.loads(z.read("pigeon.json"))
        entry.update({"id": m.get("id"), "name": m.get("name"), "kind": m.get("kind"),
                      "base_file": (m.get("base_model") or {}).get("file")})
    except (KeyError, zipfile.BadZipFile, OSError, ValueError):
        entry.update({"id": pigeon.stem, "name": pigeon.stem, "kind": None, "base_file": None})
    return entry


#: Why a skill pack in a nest is not added to the lineup: a general Reader
#: refuses the kind by name and keeps the rest (§14, §15.3). Pigeon's own
#: wording (`check.rs`), so both apps say the same thing.
SKILLPACK_IN_NEST = "a skill pack is imported as Patch, not as a Pigeon"

#: The strict entry-name rule, one path segment (the reader brief §1).
_SEGMENT_OK = __import__("re").compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def member_name(stem: str, taken: set[str]) -> str:
    """A nest member's entry name: `<slug>.pigeon`.

    NEVER the file name as given. Pigeon applies its strict zip rules to the nest
    itself, so a member called `Coin Sorter.pigeon` or `Café.pigeon` fails the
    container and refuses the WHOLE nest, and `Loud.PIGEON` is not a member at all
    because Pigeon's match is case-sensitive. `taken` holds names already used,
    lower-cased, because two names equal after lower-casing are also refused."""
    re = __import__("re")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "pigeon"
    if not _SEGMENT_OK.fullmatch(slug):
        slug = "pigeon-" + slug
    slug = slug[:240]
    arc, n = f"{slug}.pigeon", 2
    while arc.lower() in taken:
        arc = f"{slug}-{n}.pigeon"; n += 1
    return arc


def _pack_needs(p: Path) -> list[str]:
    """D84: the ids a skill pack's code asks, from its own manifest."""
    with zipfile.ZipFile(p) as z:
        needs = json.loads(z.read("pigeon.json")).get("needs") or []
    return [n for n in needs if isinstance(n, str)]


def pack(out: Path | str, pigeons, name: str | None = None) -> dict:
    """Write `<out>` from the given `.pigeon` paths. Returns nest.json.

    A sealed skill pack may travel beside the Pigeons it needs (D84): that is
    how Patch ships. Pigeon opens such a nest as Patch only when it carries one
    pack and every Pigeon the pack names, so a nest that would be refused there
    is not written here."""
    out = Path(out)
    paths = [Path(p) for p in pigeons]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"not found: {', '.join(missing)}")
    if not paths:
        raise ValueError("a nest needs at least one Pigeon")
    from . import seal as seal_mod
    seen: set[str] = set()
    manifest = {"format_version": NEST_FORMAT_VERSION, "name": name or out.stem,
                "created": date.today().isoformat(), "pigeons": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    described = [(p, _describe(p)) for p in paths]
    packs = [(p, e) for p, e in described if e["kind"] == "skillpack"]
    if len(packs) > 1:
        raise ValueError(f"{', '.join(p.name for p, _ in packs)}: a nest carries one skill pack at most")
    for p, e in packs:
        with zipfile.ZipFile(p) as zp:
            sealed = seal_mod.SEAL_NAME in zp.namelist()
        if not sealed:
            raise ValueError(f"{p.name}: a skill pack carries code, so it travels sealed — seal it first")
        ids = {x.get("id") for _, x in described}
        missing = [n for n in _pack_needs(p) if n not in ids]
        if missing:
            raise ValueError(f"{p.name} needs {', '.join(missing)}, which this nest does not carry")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for p, entry in described:
            try:
                seal_mod.strict_entries(p)
            except seal_mod.SealError as e:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"{p.name} fails the strict zip rules and would be refused: {e.reason}")
            arc = member_name(entry.get("id") or p.stem, seen)
            seen.add(arc.lower())
            z.write(p, arc)
            entry["file"] = arc
            manifest["pigeons"].append(entry)
        z.writestr("nest.json", json.dumps(manifest, indent=2) + "\n")
    # The packager's own gate, for the container: re-read what was written with
    # the strict reader every app uses. A nest that fails it is not kept.
    try:
        seal_mod.strict_entries(tmp)
    except seal_mod.SealError as e:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"the nest written fails the strict zip rules: {e.reason}")
    tmp.rename(out)
    return manifest


def list_nest(path: Path | str) -> dict:
    """nest.json, plus the members actually present (they can disagree, and
    the members are what count)."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise NotANest(f"{path.name} is not a .nest — it is not a ZIP archive", "not_a_zip")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "nest.json" not in names:
            raise NotANest(f"{path.name} has no nest.json — a folder of Pigeons zipped by hand? Add each .pigeon on its own")
        try:
            m = json.loads(z.read("nest.json"))
        except ValueError as e:
            raise NotANest(f"nest.json is not valid JSON: {e}")
    v = m.get("format_version")
    if isinstance(v, bool) or not isinstance(v, int):
        raise NotANest("nest.json has no whole-number format_version")
    if v > NEST_FORMAT_VERSION:
        raise NotANest(f"this file is .nest format {v!r}; this app reads up to {NEST_FORMAT_VERSION}", "newer_format")
    # Members exactly as Pigeon takes them: entries ending in LOWERCASE `.pigeon`
    # (its match is case-sensitive), read out of a container that passed the
    # strict rules — a bad member NAME fails the whole nest, because it fails the
    # container's own rules (the reader brief §7).
    from . import seal as seal_mod
    try:
        strict = seal_mod.strict_entries(path)
    except seal_mod.SealError as e:
        raise NotANest(f"{path.name} refused: {e.reason}", "bad_zip")
    m["members"] = [n for n, _ in strict if "/" not in n and n.endswith(".pigeon")]
    if not m["members"]:
        raise NotANest("no .pigeon files in this nest")
    return m


def unpack(path: Path | str, into: Path | str | None = None) -> list[Path]:
    """Extract the member Pigeons to a folder (a temp one by default) and
    return their paths, in the nest's own order. Only root-level members whose
    name ends in lower-case `.pigeon` are taken (spec D50)."""
    m = list_nest(path)
    into = Path(into) if into else Path(tempfile.mkdtemp(prefix="nest-"))
    into.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(path) as z:
        for name in m["members"]:
            dest = into / Path(name).name
            dest.write_bytes(z.read(name))
            out.append(dest)
    return out


def cleanup(paths) -> None:
    """Remove unpacked temporaries and their folder if it is now empty."""
    parents = set()
    for p in paths:
        p = Path(p)
        parents.add(p.parent)
        try:
            p.unlink()
        except OSError:
            pass
    for d in parents:
        try:
            if d.name.startswith("nest-") and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
