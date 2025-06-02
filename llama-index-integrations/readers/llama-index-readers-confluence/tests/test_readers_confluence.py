from unittest.mock import patch

import pytest
from llama_index.readers.confluence import ConfluenceReader


class MockConfluence:
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture(autouse=True)
def mock_atlassian_confluence(monkeypatch):
    monkeypatch.setattr("llama_index.readers.confluence", MockConfluence)


def test_confluence_reader_with_oauth2():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        oauth2={
            "client_id": "example_client_id",
            "token": {"access_token": "example_token", "token_type": "Bearer"},
        },
    )
    assert reader.confluence is not None


def test_confluence_reader_with_api_token():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        api_token="example_api_token",
    )
    assert reader.confluence is not None


def test_confluence_reader_with_cookies():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        cookies={"key": "value"},
    )
    assert reader.confluence is not None


def test_confluence_reader_with_client_args():
    with patch("atlassian.Confluence") as MockConstructor:
        reader = ConfluenceReader(
            base_url="https://example.atlassian.net/wiki",
            api_token="example_api_token",
            client_args={"backoff_and_retry": True},
        )
        assert reader.confluence is not None
        MockConstructor.assert_called_once_with(
            url="https://example.atlassian.net/wiki",
            token="example_api_token",
            cloud=True,
            backoff_and_retry=True,
        )


def test_confluence_reader_with_basic_auth():
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
        user_name="example_user",
        password="example_password",
    )
    assert reader.confluence is not None


def test_confluence_reader_with_env_api_token(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env_api_token")
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
    )
    assert reader.confluence is not None
    monkeypatch.delenv("CONFLUENCE_API_TOKEN")


def test_confluence_reader_with_env_basic_auth(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_USERNAME", "env_user")
    monkeypatch.setenv("CONFLUENCE_PASSWORD", "env_password")
    reader = ConfluenceReader(
        base_url="https://example.atlassian.net/wiki",
    )
    assert reader.confluence is not None
    monkeypatch.delenv("CONFLUENCE_USERNAME")
    monkeypatch.delenv("CONFLUENCE_PASSWORD")


def test_confluence_reader_without_credentials():
    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(base_url="https://example.atlassian.net/wiki")
    assert "Must set one of environment variables" in str(excinfo.value)


def test_confluence_reader_with_incomplete_basic_auth():
    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", user_name="example_user"
        )
    assert "Must set one of environment variables" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        ConfluenceReader(
            base_url="https://example.atlassian.net/wiki", password="example_password"
        )
    assert "Must set one of environment variables" in str(excinfo.value)
