from unittest.mock import patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.unirate import UnirateToolSpec


def _spec_with_mock_client():
    """Build a UnirateToolSpec whose underlying UnirateClient is mocked (no network)."""
    with patch("unirate.UnirateClient") as mock_client_cls:
        instance = mock_client_cls.return_value
        spec = UnirateToolSpec(api_key="test-key")
    return spec, instance


def test_class_is_tool_spec():
    names = [c.__name__ for c in UnirateToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names


def test_spec_functions():
    assert UnirateToolSpec.spec_functions == [
        "get_exchange_rate",
        "convert_currency",
        "list_supported_currencies",
        "get_vat_rates",
    ]


def test_get_exchange_rate():
    spec, client = _spec_with_mock_client()
    client.get_rate.return_value = 0.85
    assert spec.get_exchange_rate("USD", "EUR") == 0.85
    client.get_rate.assert_called_once_with(from_currency="USD", to_currency="EUR")


def test_convert_currency():
    spec, client = _spec_with_mock_client()
    client.convert.return_value = 85.0
    assert spec.convert_currency(100, "USD", "EUR") == 85.0
    client.convert.assert_called_once_with(
        to_currency="EUR", amount=100, from_currency="USD"
    )


def test_list_supported_currencies():
    spec, client = _spec_with_mock_client()
    client.get_supported_currencies.return_value = ["USD", "EUR", "GBP"]
    assert spec.list_supported_currencies() == ["USD", "EUR", "GBP"]
    client.get_supported_currencies.assert_called_once_with()


def test_get_vat_rates():
    spec, client = _spec_with_mock_client()
    client.get_vat_rates.return_value = {"country": "DE", "standard_rate": 19}
    assert spec.get_vat_rates("DE") == {"country": "DE", "standard_rate": 19}
    client.get_vat_rates.assert_called_once_with(country="DE")


def test_to_tool_list():
    spec, _ = _spec_with_mock_client()
    tools = spec.to_tool_list()
    assert len(tools) == len(UnirateToolSpec.spec_functions)
    tool_names = {t.metadata.name for t in tools}
    assert "get_exchange_rate" in tool_names
    assert "convert_currency" in tool_names
