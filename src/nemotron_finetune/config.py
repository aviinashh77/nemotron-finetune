"""Configuration loading, merging, resolution and validation.

A run config is assembled by deep-merging, in order:

    1. ``configs/base.yaml``       (all defaults live here)
    2. the user-supplied config    (``--config path/to/foo.yaml``)
    3. CLI dotlist overrides       (``training.learning_rate=1e-4``)

The merged config is then resolved (``${...}`` interpolation) and validated.
Using ``base.yaml`` as the single source of defaults keeps individual run
configs small and guarantees every field the code reads is always present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "base.yaml"

VALID_MODES = {"cpt", "dapt", "sft"}
VALID_SIZES = {"nano", "super", "ultra"}
VALID_DATA_FORMATS = {"text", "chat", "prompt_completion"}
VALID_WANDB_MODES = {"online", "offline", "disabled"}

# CPT and DAPT share the causal-LM (packed next-token) data/training path; they
# differ only in *which corpus* and *which hyper-parameters* their configs set.
MODE_TO_TASK = {"cpt": "clm", "dapt": "clm", "sft": "sft"}


class ConfigError(ValueError):
    """Raised when a run config fails validation."""


def _resolve_env(cfg: DictConfig) -> None:
    """Register a resolver so configs can reference ${oc.env:VAR,default}."""
    if not OmegaConf.has_resolver("oc.env"):
        # OmegaConf ships this resolver by default; nothing to do.
        pass


def load_config(
    config_path: str | os.PathLike,
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load, merge and validate a run configuration.

    Args:
        config_path: path to a run YAML (may be relative to ``configs/``).
        overrides: dotlist overrides, e.g. ``["training.max_steps=5"]``.

    Returns:
        A fully-resolved, validated ``DictConfig``.
    """
    _resolve_env(None)

    if not BASE_CONFIG.exists():
        raise ConfigError(f"base config not found at {BASE_CONFIG}")

    config_path = Path(config_path)
    if not config_path.is_absolute() and not config_path.exists():
        candidate = CONFIG_DIR / config_path
        if candidate.exists():
            config_path = candidate
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    base = OmegaConf.load(BASE_CONFIG)
    user = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base, user)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))

    # Record where this config came from (useful in logs / reproducibility).
    merged.run.config_path = str(config_path)

    OmegaConf.resolve(merged)
    validate_config(merged)
    return merged


def validate_config(cfg: DictConfig) -> None:
    """Validate required fields, enums and cross-field constraints."""
    errors: List[str] = []

    def require(path: str):
        node = cfg
        for part in path.split("."):
            if node is None or part not in node:
                errors.append(f"missing required field: {path}")
                return None
            node = node[part]
        if node is None:
            errors.append(f"field must not be null: {path}")
        return node

    mode = require("run.mode")
    require("run.name")
    require("run.output_dir")
    require("model.path")
    require("data.train_path")

    if mode is not None and mode not in VALID_MODES:
        errors.append(f"run.mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    size = cfg.get("model", {}).get("size")
    if size is not None and size not in VALID_SIZES:
        errors.append(f"model.size must be one of {sorted(VALID_SIZES)}, got {size!r}")

    fmt = cfg.get("data", {}).get("format")
    if fmt is not None and fmt not in VALID_DATA_FORMATS:
        errors.append(f"data.format must be one of {sorted(VALID_DATA_FORMATS)}, got {fmt!r}")

    wmode = cfg.get("wandb", {}).get("mode")
    if wmode is not None and wmode not in VALID_WANDB_MODES:
        errors.append(f"wandb.mode must be one of {sorted(VALID_WANDB_MODES)}, got {wmode!r}")

    # Mode / data-format coherence.
    if mode in {"cpt", "dapt"} and fmt not in {"text", None}:
        errors.append(
            f"run.mode={mode!r} expects data.format='text' (packed causal LM), got {fmt!r}"
        )
    if mode == "sft" and fmt not in {"chat", "prompt_completion", None}:
        errors.append(
            f"run.mode='sft' expects data.format in {{'chat','prompt_completion'}}, got {fmt!r}"
        )

    lora = cfg.get("lora", {})
    if lora.get("enabled", True):
        if int(lora.get("r", 0)) <= 0:
            errors.append("lora.r must be a positive integer when lora.enabled")
        tm = lora.get("target_modules")
        if not (tm == "all-linear" or (hasattr(tm, '__len__') and len(tm) > 0)):
            errors.append("lora.target_modules must be 'all-linear' or a non-empty list")

    tr = cfg.get("training", {})
    if int(tr.get("max_steps", -1)) <= 0 and float(tr.get("num_train_epochs", 0)) <= 0:
        errors.append("training: set either max_steps>0 or num_train_epochs>0")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def to_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def save_resolved(cfg: DictConfig, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(cfg))
