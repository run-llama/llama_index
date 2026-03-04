import pytest
from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.dappier import DappierRealTimeSearchToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in DappierRealTimeSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@pytest.fixture()
def dappier_client():
    return MagicMock()


@pytest.fixture()
def tool(dappier_client):
    tool_instance = DappierRealTimeSearchToolSpec(api_key="your-api-key")
    tool_instance.client = dappier_client
    return tool_instance


class TestDappierRealTimeSearchTool:
    def test_init_without_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("DAPPIER_API_KEY", raising=False)
        dappier_client = MagicMock()
        with patch("dappier.Dappier", return_value=dappier_client):
            with pytest.raises(ValueError) as excinfo:
                DappierRealTimeSearchToolSpec()
        assert "API key is required" in str(excinfo.value)

    def test_search_real_time_data_returns_response_message(self, tool, dappier_client):
        response = MagicMock()
        response.message = "Real-time data result"
        dappier_client.search_real_time_data.return_value = response

        result = tool.search_real_time_data("test query")
        assert result == "Real-time data result"
        dappier_client.search_real_time_data.assert_called_once_with(
            query="test query", ai_model_id="am_01j0rzq4tvfscrgzwac7jv1p4c"
        )

    def test_search_stock_market_data_returns_response_message(
        self, tool, dappier_client
    ):
        response = MagicMock()
        response.message = "Stock market data result"
        dappier_client.search_real_time_data.return_value = response

        result = tool.search_stock_market_data("stock query")
        assert result == "Stock market data result"
        dappier_client.search_real_time_data.assert_called_once_with(
            query="stock query", ai_model_id="am_01j749h8pbf7ns8r1bq9s2evrh"
        )

    def test_search_real_time_data_no_response(self, tool, dappier_client):
        dappier_client.search_real_time_data.return_value = None

        result = tool.search_real_time_data("test query")
        assert result == "No real-time data found."

    def test_search_stock_market_data_no_response(self, tool, dappier_client):
        dappier_client.search_real_time_data.return_value = None

        result = tool.search_stock_market_data("stock query")
        assert result == "No stock market data found."
