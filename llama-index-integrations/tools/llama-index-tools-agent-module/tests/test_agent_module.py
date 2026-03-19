"""Tests for AgentModuleToolSpec."""

from unittest.mock import MagicMock, patch


from llama_index.tools.agent_module.base import AgentModuleToolSpec


def test_tool_spec_init_defaults():
    spec = AgentModuleToolSpec()
    assert spec.am_key is None
    assert spec.vertical == "ethics"
    assert spec.timeout == 10


def test_tool_spec_init_with_key():
    spec = AgentModuleToolSpec(am_key="am_test_key_123")
    assert spec.am_key == "am_test_key_123"


def test_build_headers_with_key():
    spec = AgentModuleToolSpec(am_key="am_test_key_123")
    headers = spec._build_headers()
    assert headers == {"X-Agent-Module-Key": "am_test_key_123"}


def test_build_headers_without_key():
    spec = AgentModuleToolSpec()
    headers = spec._build_headers()
    assert headers == {}


def test_to_node_id():
    spec = AgentModuleToolSpec()
    assert spec._to_node_id("ETH_013", "ethics") == "node:ethics:eth013"
    assert spec._to_node_id("ETH_021", "ethics") == "node:ethics:eth021"
    assert spec._to_node_id("ETH_016", "travel") == "node:travel:eth016"


def test_spec_functions():
    spec = AgentModuleToolSpec()
    expected = [
        "query_module",
        "query_fria",
        "query_prohibited_practices",
        "query_high_risk_classification",
        "query_risk_management",
        "query_conformity_assessment",
        "query_gpai_obligations",
    ]
    assert spec.spec_functions == expected


def test_to_tool_list():
    spec = AgentModuleToolSpec(am_key="am_test_key_123")
    tools = spec.to_tool_list()
    assert len(tools) == len(spec.spec_functions)


@patch("llama_index.tools.agent_module.base.requests.get")
def test_query_module(mock_get):
    mock_response = MagicMock()
    mock_response.text = '{"module": "ETH_013", "records": []}'
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    spec = AgentModuleToolSpec(am_key="am_test_key_123")
    result = spec.query_module("ETH_013")

    assert result == '{"module": "ETH_013", "records": []}'
    mock_get.assert_called_once_with(
        "https://api.agent-module.dev/api/demo",
        params={"vertical": "ethics", "node": "node:ethics:eth013"},
        headers={"X-Agent-Module-Key": "am_test_key_123"},
        timeout=10,
    )


@patch("llama_index.tools.agent_module.base.requests.get")
def test_query_fria(mock_get):
    mock_response = MagicMock()
    mock_response.text = '{"module": "ETH_021"}'
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    spec = AgentModuleToolSpec(am_key="am_test_key_123")
    result = spec.query_fria()

    assert "ETH_021" in result
    call_args = mock_get.call_args
    assert call_args[1]["params"]["node"] == "node:ethics:eth021"


@patch("llama_index.tools.agent_module.base.requests.get")
def test_query_high_risk_no_key(mock_get):
    mock_response = MagicMock()
    mock_response.text = '{"module": "ETH_015"}'
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    spec = AgentModuleToolSpec()
    result = spec.query_high_risk_classification()

    call_args = mock_get.call_args
    assert call_args[1]["headers"] == {}
    assert call_args[1]["params"]["node"] == "node:ethics:eth015"


@patch("llama_index.tools.agent_module.base.requests.get")
def test_http_error_handling(mock_get):
    import requests as req

    mock_response = MagicMock()
    mock_response.status_code = 401
    error = req.HTTPError(response=mock_response)
    mock_get.return_value.raise_for_status.side_effect = error

    spec = AgentModuleToolSpec(am_key="bad_key")
    result = spec.query_fria()

    assert "HTTP error" in result
    assert "401" in result


@patch("llama_index.tools.agent_module.base.requests.get")
def test_connection_error_handling(mock_get):
    import requests as req

    mock_get.side_effect = req.ConnectionError("Connection refused")

    spec = AgentModuleToolSpec()
    result = spec.query_prohibited_practices()

    assert "connection error" in result.lower()


@patch("llama_index.tools.agent_module.base.requests.get")
def test_custom_vertical(mock_get):
    mock_response = MagicMock()
    mock_response.text = '{"vertical": "travel"}'
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    spec = AgentModuleToolSpec(am_key="am_test_key_123")
    spec.query_module("TRAVEL_001", vertical="travel")

    call_args = mock_get.call_args
    assert call_args[1]["params"]["vertical"] == "travel"
    assert call_args[1]["params"]["node"] == "node:travel:travel001"
