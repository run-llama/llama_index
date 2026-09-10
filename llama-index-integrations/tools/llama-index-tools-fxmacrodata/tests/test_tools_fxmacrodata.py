import urllib.error
from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.fxmacrodata import FXMacroDataToolSpec

PAYLOAD = '{"currency":"USD","data":[]}'


def test_class():
    names_of_base_classes = [b.__name__ for b in FXMacroDataToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_every_spec_function_exists():
    spec = FXMacroDataToolSpec()
    for name in spec.spec_functions:
        assert callable(getattr(spec, name)), f"{name} is listed but not implemented"


def test_to_tool_list_exposes_all_functions():
    spec = FXMacroDataToolSpec()
    assert len(spec.to_tool_list()) == len(spec.spec_functions)


def _mock_urlopen(captured, body=PAYLOAD):
    def _open(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        response = MagicMock()
        response.read.return_value = body.encode("utf-8")
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *args: None
        return response

    return _open


def test_currency_is_lowercased_into_the_path():
    captured = {}
    spec = FXMacroDataToolSpec(api_key="test_key")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        spec.get_latest_macro_snapshot("JPY")

    assert captured["url"] == "https://api.fxmacrodata.com/v1/announcements/jpy/latest"


def test_unset_optional_dates_are_not_sent():
    captured = {}
    spec = FXMacroDataToolSpec(api_key="test_key")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        spec.get_indicator_history(currency="USD", indicator="gdp", limit=5)

    assert "limit=5" in captured["url"]
    assert "start_date" not in captured["url"]
    assert "end_date" not in captured["url"]


def test_api_key_is_a_header_not_a_query_parameter():
    captured = {}
    spec = FXMacroDataToolSpec(api_key="test_key")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        spec.get_market_sessions()

    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["x-api-key"] == "test_key"
    assert "test_key" not in captured["url"]


def test_no_auth_header_without_a_key(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    captured = {}
    spec = FXMacroDataToolSpec()
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        spec.get_risk_sentiment()

    assert "x-api-key" not in {k.lower() for k in captured["headers"]}


def test_auth_failure_explains_the_key_requirement():
    spec = FXMacroDataToolSpec(api_key="test_key")
    error = urllib.error.HTTPError("url", 403, "Forbidden", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        result = spec.get_cot_positioning("GBP")

    assert "requires an API key" in result
    assert "USD" in result


def test_server_error_is_not_reported_as_an_auth_problem():
    spec = FXMacroDataToolSpec(api_key="test_key")
    error = urllib.error.HTTPError("url", 500, "Server Error", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        result = spec.get_commodity_prices()

    assert "HTTP 500" in result
    assert "API key" not in result


def test_network_error_is_returned_not_raised():
    spec = FXMacroDataToolSpec(api_key="test_key")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        result = spec.get_market_sessions()

    assert "failed" in result


def test_non_http_base_url_is_refused():
    # base_url is settable, so a file:// base must not reach urlopen.
    spec = FXMacroDataToolSpec(api_key="test_key", base_url="file:///etc")
    result = spec.get_market_sessions()

    assert "unsupported URL scheme" in result


def test_base_url_trailing_slash_does_not_double():
    captured = {}
    spec = FXMacroDataToolSpec(api_key="k", base_url="https://example.test/v1/")
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        spec.get_market_sessions()

    assert captured["url"] == "https://example.test/v1/market_sessions"
