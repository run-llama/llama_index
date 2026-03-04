import pytest
from llama_index.embeddings.openai.utils import (
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_OPENAI_API_VERSION,
    MISSING_API_KEY_ERROR_MESSAGE,
    resolve_openai_credentials,
    validate_openai_api_key,
)


def test_validate_openai_api_key_with_valid_key() -> None:
    validate_openai_api_key("valid_api_key")


def test_validate_openai_api_key_with_env_var(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "valid_api_key")
    validate_openai_api_key()


def test_validate_openai_api_key_with_no_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match=MISSING_API_KEY_ERROR_MESSAGE):
        validate_openai_api_key()


def test_validate_openai_api_key_with_empty_env_var(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match=MISSING_API_KEY_ERROR_MESSAGE):
        validate_openai_api_key()


def test_resolve_openai_credentials_with_params() -> None:
    api_key, api_base, api_version = resolve_openai_credentials(
        api_key="param_api_key",
        api_base="param_api_base",
        api_version="param_api_version",
    )
    assert api_key == "param_api_key"
    assert api_base == "param_api_base"
    assert api_version == "param_api_version"


def test_resolve_openai_credentials_with_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_api_key")
    monkeypatch.setenv("OPENAI_API_BASE", "env_api_base")
    monkeypatch.setenv("OPENAI_API_VERSION", "env_api_version")
    api_key, api_base, api_version = resolve_openai_credentials()
    assert api_key == "env_api_key"
    assert api_base == "env_api_base"
    assert api_version == "env_api_version"


def test_resolve_openai_credentials_with_openai_module(monkeypatch) -> None:
    monkeypatch.setattr("openai.base_url", "module_api_base")
    monkeypatch.setattr("openai.api_version", "module_api_version")
    api_key, api_base, api_version = resolve_openai_credentials()
    assert api_base == "module_api_base"
    assert api_version == "module_api_version"


def test_resolve_openai_credentials_with_defaults() -> None:
    api_key, api_base, api_version = resolve_openai_credentials()
    assert api_base == DEFAULT_OPENAI_API_BASE
    assert api_version == DEFAULT_OPENAI_API_VERSION
