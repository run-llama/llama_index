from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.verifly import VeriflyToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in VeriflyToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert "verify_email" in VeriflyToolSpec.spec_functions


def test_init_with_api_key():
    tool = VeriflyToolSpec(api_key="vf_test")
    assert tool.api_key == "vf_test"
    assert tool.base_url == "https://verifly.email"


def test_init_from_env(monkeypatch):
    monkeypatch.setenv("VERIFLY_API_KEY", "vf_env")
    tool = VeriflyToolSpec()
    assert tool.api_key == "vf_env"


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("VERIFLY_API_KEY", raising=False)
    with pytest.raises(ValueError):
        VeriflyToolSpec()


def test_base_url_trailing_slash():
    tool = VeriflyToolSpec(api_key="vf_test", base_url="https://example.com/")
    assert tool.base_url == "https://example.com"


@patch("llama_index.tools.verifly.base.requests.get")
def test_verify_email(mock_get):
    mock_response = Mock()
    mock_response.json.return_value = {
        "success": True,
        "email": "lead@example.com",
        "is_valid": True,
        "result": "deliverable",
        "reason": "Mailbox exists",
        "recommendation": "send",
        "credits": {"used": 1, "remaining": 99},
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    tool = VeriflyToolSpec(api_key="vf_test")
    doc = tool.verify_email("lead@example.com")

    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"] == {"email": "lead@example.com"}
    assert kwargs["headers"]["Authorization"] == "Bearer vf_test"

    assert isinstance(doc, Document)
    assert "deliverable" in doc.text
    assert "recommendation: send" in doc.text
    assert doc.metadata["is_valid"] is True
    assert doc.metadata["result"] == "deliverable"
