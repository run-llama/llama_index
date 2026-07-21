import json
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.haunt import HauntToolSpec


def _result(success, data=None, error_code=None, message=None):
    result = Mock()
    result.success = success
    result.data = data
    result.error_code = error_code
    result.message = message
    result.error = None
    return result


def test_class():
    names_of_base_classes = [b.__name__ for b in HauntToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert "extract" in HauntToolSpec.spec_functions
    assert "load" in HauntToolSpec.spec_functions


@patch("hauntapi.Haunt")
def test_init(mock_haunt):
    tool = HauntToolSpec(api_key="test_api_key")
    mock_haunt.assert_called_once_with(api_key="test_api_key")
    assert tool.client == mock_haunt.return_value


@patch("hauntapi.Haunt")
def test_extract_success_returns_json(mock_haunt):
    client = Mock()
    client.extract.return_value = _result(True, data={"title": "Example Domain"})
    mock_haunt.return_value = client

    tool = HauntToolSpec(api_key="test_key")
    out = tool.extract("https://example.com", "the page title")

    client.extract.assert_called_once_with("https://example.com", "the page title")
    assert json.loads(out) == {"title": "Example Domain"}


@patch("hauntapi.Haunt")
def test_extract_honest_failure_returns_error_code(mock_haunt):
    client = Mock()
    client.extract.return_value = _result(
        False, error_code="login_required", message="This page needs a login."
    )
    mock_haunt.return_value = client

    tool = HauntToolSpec(api_key="test_key")
    out = json.loads(tool.extract("https://example.com/admin", "the numbers"))

    assert out["error_code"] == "login_required"
    assert "login" in out["message"].lower()


@patch("hauntapi.Haunt")
def test_load_returns_markdown_documents(mock_haunt):
    client = Mock()
    client.extract.side_effect = [
        _result(True, data={"markdown": "# Page A"}),
        _result(True, data="# Page B"),
    ]
    mock_haunt.return_value = client

    tool = HauntToolSpec(api_key="test_key")
    docs = tool.load(["https://a.example", "https://b.example"])

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert docs[0].text == "# Page A"
    assert docs[0].extra_info["url"] == "https://a.example"
    assert docs[1].text == "# Page B"


@patch("hauntapi.Haunt")
def test_load_single_url_string(mock_haunt):
    client = Mock()
    client.extract.return_value = _result(True, data={"markdown": "# Page"})
    mock_haunt.return_value = client

    tool = HauntToolSpec(api_key="test_key")
    docs = tool.load("https://a.example")

    assert len(docs) == 1


@patch("hauntapi.Haunt")
def test_load_unreadable_page_raises_never_invents(mock_haunt):
    client = Mock()
    client.extract.return_value = _result(
        False, error_code="access_denied", message="Bot wall."
    )
    mock_haunt.return_value = client

    tool = HauntToolSpec(api_key="test_key")
    with pytest.raises(ValueError, match="access_denied"):
        tool.load(["https://a.example"])
