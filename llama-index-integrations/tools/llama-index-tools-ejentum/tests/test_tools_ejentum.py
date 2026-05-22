"""Tests for the Ejentum Reasoning Harness tool spec."""

import os

import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.ejentum import EjentumToolSpec
from llama_index.tools.ejentum.base import DEFAULT_API_URL, SUPPORTED_MODES
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


def test_class_is_subclass_of_mcp_tool_spec() -> None:
    assert issubclass(EjentumToolSpec, McpToolSpec)
    assert issubclass(EjentumToolSpec, BaseToolSpec)


def test_init_with_explicit_api_key() -> None:
    spec = EjentumToolSpec(api_key="test-key")
    assert isinstance(spec.client, BasicMCPClient)
    assert spec.client.command_or_url == DEFAULT_API_URL
    assert spec.client.headers == {"Authorization": "Bearer test-key"}
    assert spec.allowed_tools is None


def test_init_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EJENTUM_API_KEY", "env-key")
    spec = EjentumToolSpec()
    assert spec.client.headers == {"Authorization": "Bearer env-key"}


def test_init_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EJENTUM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="EJENTUM_API_KEY"):
        EjentumToolSpec()


def test_modes_subset_filters_allowed_tools() -> None:
    spec = EjentumToolSpec(api_key="k", modes=["reasoning", "code"])
    assert spec.allowed_tools == ["harness_reasoning", "harness_code"]


def test_modes_unknown_value_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        EjentumToolSpec(api_key="k", modes=["reasoning", "nonexistent"])


def test_custom_api_url_and_timeout() -> None:
    spec = EjentumToolSpec(
        api_key="k",
        api_url="https://example.com/mcp",
        timeout=10,
    )
    assert spec.client.command_or_url == "https://example.com/mcp"
    assert spec.client.timeout == 10


def test_supported_modes_constant_matches_four_harnesses() -> None:
    assert set(SUPPORTED_MODES) == {
        "reasoning",
        "code",
        "anti_deception",
        "memory",
    }
