"""Path discovery and validation.

Nothing in this package hardcodes a filesystem location. Everything comes from
config.toml, expanded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# tomllib is standard from Python 3.11. The Mac this was written on has 3.11;
# the Mac the pack was first carried to has the 3.9.6 that ships with the
# Xcode Command Line Tools, and the very first run died here. `tomli` is the
# same module under its pre-3.11 name; requirements-train.txt installs it
# where it is needed.
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "this Python is older than 3.11 and has no `tomli` — run the training-tools "
            "setup again (it installs tomli), or install Python 3.11 or newer"
        ) from e
from pathlib import Path

# VENDORED into Flight School (from the pilot repo, the
# Instructor pilot). One deliberate patch, and only one: ROOT — which anchors
# the derived runs/plans/briefs/corpora directories and CONFIG_PATH — can be
# supplied by the host through INSTRUCTOR_ROOT, because inside an app bundle
# "next to the package" is read-only. Everything else is byte-identical to the
# source repo; fixes belong THERE first and get re-vendored.
ROOT = Path(os.environ.get("INSTRUCTOR_ROOT") or Path(__file__).resolve().parent.parent)
CONFIG_PATH = ROOT / "config.toml"


class ConfigError(RuntimeError):
    pass


def _expand(value: str) -> Path:
    """~ and $VARS, resolved against the project root for relative paths."""
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (ROOT / p)


@dataclass(frozen=True)
class Config:
    llama_server: Path
    llama_tokenize: Path
    instructor_gguf: Path
    target_gguf: Path
    target_mmproj: Path
    target_repo: str
    instructor_repo: str
    hf_home: Path
    venv_train: Path
    venv_convert: Path
    venv_vision: Path
    mlx_to_peft: Path
    lora_to_gguf: Path
    # the tiny demo base (two-character chat). Optional tier: absent entries
    # must not break the pilot.
    demo_gguf: Path | None
    demo_repo: str | None
    generation_budget_gb: float
    total_ram_gb: float

    # -- derived locations -------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return ROOT / "runs"

    def train_env(self) -> dict[str, str]:
        """Environment for any subprocess that touches the HF cache.

        HF_HOME is pointed at Flight School's cache so the 4-bit weights it
        already downloaded are reused rather than fetched a second time.
        """
        env = dict(os.environ)
        env["HF_HOME"] = str(self.hf_home)
        return env


def load(path: Path | None = None) -> Config:
    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config.toml not found at {cfg_path}")
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    def get(section: str, key: str) -> str:
        try:
            return raw[section][key]
        except KeyError as exc:
            raise ConfigError(f"config.toml is missing [{section}].{key}") from exc

    return Config(
        llama_server=_expand(get("binaries", "llama_server")),
        llama_tokenize=_expand(get("binaries", "llama_tokenize")),
        instructor_gguf=_expand(get("models", "instructor_gguf")),
        target_gguf=_expand(get("models", "target_gguf")),
        target_mmproj=_expand(get("models", "target_mmproj")),
        target_repo=get("mlx", "target_repo"),
        instructor_repo=get("mlx", "instructor_repo"),
        hf_home=_expand(get("mlx", "hf_home")),
        venv_train=_expand(get("venvs", "train")),
        venv_convert=_expand(get("venvs", "convert")),
        venv_vision=_expand(get("venvs", "vision")),
        mlx_to_peft=_expand(get("vendor", "mlx_to_peft")),
        lora_to_gguf=_expand(get("vendor", "lora_to_gguf")),
        demo_gguf=_expand(raw["demo"]["demo_gguf"]) if "demo" in raw else None,
        demo_repo=raw.get("demo", {}).get("demo_repo"),
        generation_budget_gb=float(raw["limits"]["generation_budget_gb"]),
        total_ram_gb=float(raw["limits"]["total_ram_gb"]),
    )


def _mlx_cached(cfg: Config, repo: str) -> bool:
    """Is this MLX repo already in the HF cache with real weights?"""
    slug = "models--" + repo.replace("/", "--")
    snaps = cfg.hf_home / "hub" / slug / "snapshots"
    if not snaps.is_dir():
        return False
    return any(any(s.glob("*.safetensors")) for s in snaps.iterdir() if s.is_dir())
