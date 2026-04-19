"""Config loader.

Parses `config.yaml` into a typed `Config` object at process startup.
Precedence (per .agent/memory/gotchas.md): env vars > config file > defaults.

Env-var override is scoped: only fields that frequently vary between
machines (API keys, endpoints) are overridable today. Wider overrides can
be added when a real use case demands them.

The dataclasses are frozen so the pipeline can safely share them across
tasks. If you need to mutate something, build a new Config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

# Per-request ``PipelineRequest.tactics_override`` uses these names to *disable* tactics.
TACTIC_DISABLE_NAMES = frozenset({
    "t1_route",
    "t2_compress",
    "t3_sem_cache",
    "t4_draft",
    "t5_diff",
    "t6_intent",
    "t7_batch",
})

Backend = Literal["ollama", "openai_compat", "anthropic"]


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or malformed."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """How to reach one chat/embedding backend."""

    backend: Backend
    endpoint: str
    chat_model: str
    embed_model: str | None = None
    api_key_env: str | None = None
    num_ctx: int = 8192

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, where: str) -> ModelConfig:
        missing = [k for k in ("backend", "endpoint", "chat_model") if k not in data]
        if missing:
            raise ConfigError(f"{where}: missing required keys {missing}")
        backend = data["backend"]
        if backend not in ("ollama", "openai_compat", "anthropic"):
            raise ConfigError(f"{where}: backend must be 'ollama', 'openai_compat', or 'anthropic'")
        return cls(
            backend=backend,
            endpoint=str(data["endpoint"]),
            chat_model=str(data["chat_model"]),
            embed_model=data.get("embed_model"),
            api_key_env=data.get("api_key_env"),
            num_ctx=int(data.get("num_ctx", 8192)),
        )


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Which transports to expose."""

    mcp: bool = True
    http: bool = True
    http_host: str = "127.0.0.1"
    http_port: int = 7788

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransportConfig:
        return cls(
            mcp=bool(data.get("mcp", True)),
            http=bool(data.get("http", True)),
            http_host=str(data.get("http_host", "127.0.0.1")),
            http_port=int(data.get("http_port", 7788)),
        )


@dataclass(frozen=True, slots=True)
class TacticsConfig:
    """Which tactics are enabled. Stage 3 passthrough ignores these; Stage 4+
    uses them to decide whether to run each pipeline stage."""

    t1_route: bool = False
    t2_compress: bool = False
    t3_sem_cache: bool = False
    t4_draft: bool = False
    t5_diff: bool = False
    t6_intent: bool = False
    t7_batch: bool = False
    tools_require_cloud: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TacticsConfig:
        def enabled(key: str) -> bool:
            stage = data.get(key)
            if isinstance(stage, dict):
                return bool(stage.get("enabled", False))
            if isinstance(stage, bool):
                return stage
            return False

        params: dict[str, Any] = {}
        for key in (
            "t1_route",
            "t2_compress",
            "t3_sem_cache",
            "t4_draft",
            "t5_diff",
            "t6_intent",
            "t7_batch",
        ):
            stage = data.get(key)
            if isinstance(stage, dict):
                params[key] = {k: v for k, v in stage.items() if k != "enabled"}

        return cls(
            t1_route=enabled("t1_route"),
            t2_compress=enabled("t2_compress"),
            t3_sem_cache=enabled("t3_sem_cache"),
            t4_draft=enabled("t4_draft"),
            t5_diff=enabled("t5_diff"),
            t6_intent=enabled("t6_intent"),
            t7_batch=enabled("t7_batch"),
            tools_require_cloud=bool(data.get("tools_require_cloud", True)),
            params=params,
        )

    def any_enabled(self) -> bool:
        return any(
            (
                self.t1_route,
                self.t2_compress,
                self.t3_sem_cache,
                self.t4_draft,
                self.t5_diff,
                self.t6_intent,
                self.t7_batch,
            )
        )


def apply_tactics_override(
    tactics: TacticsConfig, override: frozenset[str] | None
) -> TacticsConfig:
    """Return a tactics view with listed tactic names forced off.

    Unknown names in ``override`` are ignored. ``None`` or empty means no change.
    """
    if not override:
        return tactics
    o = frozenset(x for x in override if x in TACTIC_DISABLE_NAMES)
    if not o:
        return tactics
    return replace(
        tactics,
        t1_route=tactics.t1_route and "t1_route" not in o,
        t2_compress=tactics.t2_compress and "t2_compress" not in o,
        t3_sem_cache=tactics.t3_sem_cache and "t3_sem_cache" not in o,
        t4_draft=tactics.t4_draft and "t4_draft" not in o,
        t5_diff=tactics.t5_diff and "t5_diff" not in o,
        t6_intent=tactics.t6_intent and "t6_intent" not in o,
        t7_batch=tactics.t7_batch and "t7_batch" not in o,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
    """Optional hints in ``/v1/splitter/stats`` when routing skews very local."""

    enabled: bool = False
    min_requests: int = 20
    max_local_fraction: float = 0.45
    hint: str = (
        "Local/cache fraction is high vs total requests. Consider enabling "
        "t1_route.verify_trivial, raising min_user_chars, or using a conservative preset."
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AdaptiveConfig:
        if not data:
            return cls()
        default = (
            "Local/cache fraction is high vs total requests. Consider enabling "
            "t1_route.verify_trivial, raising min_user_chars, or using a conservative preset."
        )
        return cls(
            enabled=bool(data.get("enabled", False)),
            min_requests=int(data.get("min_requests", 20)),
            max_local_fraction=float(data.get("max_local_fraction", 0.45)),
            hint=str(data.get("hint", default)),
        )


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level config shared across both transports and the pipeline."""

    cloud: ModelConfig | None = None
    local: ModelConfig | None = None
    transport: TransportConfig = field(default_factory=TransportConfig)
    tactics: TacticsConfig = field(default_factory=TacticsConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    log_file: Path | None = None
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ConfigError("top-level config must be a mapping")
        version = int(data.get("version", 1))
        if version != 1:
            raise ConfigError(f"unsupported config version {version}; expected 1")

        models = data.get("models") or {}
        cloud = (
            ModelConfig.from_dict(models["cloud"], where="models.cloud")
            if models.get("cloud")
            else None
        )
        local = (
            ModelConfig.from_dict(models["local"], where="models.local")
            if models.get("local")
            else None
        )
        if cloud is None and local is None:
            raise ConfigError("at least one of models.cloud or models.local is required")

        transport = TransportConfig.from_dict(data.get("transport") or {})
        tactics = TacticsConfig.from_dict(data.get("pipeline") or {})
        adaptive = AdaptiveConfig.from_dict(data.get("adaptive"))
        log_file_raw = (data.get("evaluation") or {}).get("log_file")
        log_file = Path(log_file_raw) if log_file_raw else None

        return cls(
            cloud=cloud,
            local=local,
            transport=transport,
            tactics=tactics,
            adaptive=adaptive,
            log_file=log_file,
            version=version,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text())
        except FileNotFoundError as e:
            raise ConfigError(f"config file not found: {p}") from e
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {p}: {e}") from e
        if raw is None:
            raise ConfigError(f"config file {p} is empty")
        return cls.from_dict(raw)


def load_config(path: Path | str | None = None) -> Config:
    """Resolve the config path, load, and apply env overrides.

    Resolution order:
    1. Explicit `path` argument.
    2. `LOCAL_SPLITTER_CONFIG` env var.
    3. `.local_splitter/config.yaml` in CWD.
    4. `config.yaml` in CWD.

    Raises `ConfigError` if no file is found.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("LOCAL_SPLITTER_CONFIG")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path.cwd() / ".local_splitter" / "config.yaml")
        candidates.append(Path.cwd() / "config.yaml")

    for candidate in candidates:
        if candidate.is_file():
            return Config.from_yaml(candidate)

    raise ConfigError(
        "no config file found; tried: " + ", ".join(str(c) for c in candidates)
    )


__all__ = [
    "AdaptiveConfig",
    "Backend",
    "Config",
    "ConfigError",
    "ModelConfig",
    "TACTIC_DISABLE_NAMES",
    "TacticsConfig",
    "TransportConfig",
    "apply_tactics_override",
    "load_config",
]
