"""Unit tests for `local_splitter.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_splitter.config import (
    Config,
    ConfigError,
    ModelConfig,
    TacticsConfig,
    load_config,
)


VALID_MIN = {
    "version": 1,
    "models": {
        "cloud": {
            "backend": "openai_compat",
            "endpoint": "https://api.openai.com/v1",
            "chat_model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        }
    },
}

VALID_FULL = {
    "version": 1,
    "transport": {
        "mcp": True,
        "http": True,
        "http_host": "0.0.0.0",
        "http_port": 8080,
    },
    "models": {
        "local": {
            "backend": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "chat_model": "llama3.2:3b",
            "embed_model": "nomic-embed-text",
            "num_ctx": 16384,
        },
        "cloud": {
            "backend": "openai_compat",
            "endpoint": "https://api.openai.com/v1",
            "chat_model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    },
    "pipeline": {
        "t1_route": {"enabled": True, "trivial_threshold": 0.8},
        "t3_sem_cache": {"enabled": True, "similarity_threshold": 0.92},
        "t5_diff": True,
    },
    "evaluation": {"log_file": ".local_splitter/runs.jsonl"},
}


def test_from_dict_minimum_config() -> None:
    c = Config.from_dict(VALID_MIN)
    assert c.cloud.backend == "openai_compat"
    assert c.cloud.chat_model == "gpt-4o-mini"
    assert c.local is None
    assert c.transport.http_port == 7788  # default
    assert not c.tactics.any_enabled()


def test_from_dict_full_config() -> None:
    c = Config.from_dict(VALID_FULL)
    assert c.local is not None
    assert c.local.backend == "ollama"
    assert c.local.num_ctx == 16384
    assert c.transport.http_host == "0.0.0.0"
    assert c.tactics.t1_route is True
    assert c.tactics.t3_sem_cache is True
    assert c.tactics.t5_diff is True  # bare-bool form
    assert c.tactics.t2_compress is False
    assert c.tactics.params["t1_route"]["trivial_threshold"] == 0.8
    assert c.log_file == Path(".local_splitter/runs.jsonl")


def test_missing_cloud_raises() -> None:
    with pytest.raises(ConfigError, match="models.cloud"):
        Config.from_dict({"version": 1, "models": {}})


def test_bad_backend_raises() -> None:
    bad = {
        "models": {
            "cloud": {
                "backend": "not-a-thing",
                "endpoint": "x",
                "chat_model": "y",
            }
        }
    }
    with pytest.raises(ConfigError, match="backend"):
        Config.from_dict(bad)


def test_missing_required_model_field() -> None:
    bad = {
        "models": {
            "cloud": {
                "backend": "ollama",
                "endpoint": "http://x",
                # chat_model missing
            }
        }
    }
    with pytest.raises(ConfigError, match="chat_model"):
        Config.from_dict(bad)


def test_unsupported_version() -> None:
    with pytest.raises(ConfigError, match="version"):
        Config.from_dict({"version": 99, "models": VALID_MIN["models"]})


def test_empty_yaml_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ConfigError, match="empty"):
        Config.from_yaml(p)


def test_missing_yaml_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(tmp_path / "nope.yaml")


def test_invalid_yaml_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("::: this is not yaml :::\n[[[")
    with pytest.raises(ConfigError, match="invalid YAML"):
        Config.from_yaml(p)


def test_load_config_with_explicit_path(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("version: 1\nmodels:\n  cloud:\n    backend: ollama\n    endpoint: http://x\n    chat_model: m\n")
    c = load_config(p)
    assert c.cloud.backend == "ollama"


def test_load_config_uses_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "version: 1\nmodels:\n  cloud:\n    backend: ollama\n    endpoint: http://y\n    chat_model: n\n"
    )
    monkeypatch.setenv("LOCAL_SPLITTER_CONFIG", str(p))
    # Set cwd somewhere without a config.yaml so only the env var matches.
    monkeypatch.chdir(tmp_path / "..")
    c = load_config()
    assert c.cloud.endpoint == "http://y"


def test_load_config_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCAL_SPLITTER_CONFIG", raising=False)
    with pytest.raises(ConfigError, match="no config file found"):
        load_config()


def test_tactics_any_enabled() -> None:
    off = TacticsConfig()
    assert off.any_enabled() is False
    on = TacticsConfig(t4_draft=True)
    assert on.any_enabled() is True


def test_model_config_defaults() -> None:
    mc = ModelConfig(
        backend="ollama", endpoint="http://x", chat_model="m"
    )
    assert mc.num_ctx == 8192
    assert mc.embed_model is None
    assert mc.api_key_env is None
