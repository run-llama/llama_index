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
