from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in WolframAlphaToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@patch("llama_index.tools.wolfram_alpha.base.requests.get")
def test_api_params_in_url(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.text = "result"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool_spec = WolframAlphaToolSpec(
        app_id="test-id",
        api_params={"maxchars": 1000, "units": "metric"},
    )
    tool_spec.wolfram_alpha_query("test query")

    call_url = mock_get.call_args[0][0]
    assert "input=test+query" in call_url
    assert "maxchars=1000" in call_url
    assert "units=metric" in call_url


def test_wolfram_alpha_query_sets_timeout(monkeypatch):
    import requests
    from llama_index.tools.wolfram_alpha import WolframAlphaToolSpec

    calls = []

    class _Response:
        text = "result"

        def raise_for_status(self):
            pass

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    assert WolframAlphaToolSpec(app_id="test-id").wolfram_alpha_query("2+2") == "result"
    assert calls[0].get("timeout")
