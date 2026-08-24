from typing import Any

import pytest
import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.finance import (
    AdanosMarketSentimentToolSpec,
    FinanceAgentToolSpec,
)

TEST_VALUE = "test-value"


def test_class():
    name_of_base_classes = [b.__name__ for b in FinanceAgentToolSpec.__mro__]
    assert BaseToolSpec.__name__ in name_of_base_classes


@pytest.fixture()
def adanos_tool() -> AdanosMarketSentimentToolSpec:
    return AdanosMarketSentimentToolSpec(TEST_VALUE)


def mock_response(mocker, payload: Any):
    response = mocker.Mock()
    response.json.return_value = payload
    return response


def test_adanos_tool_spec_exposes_agent_functions(adanos_tool):
    names = [tool.metadata.name for tool in adanos_tool.to_tool_list()]

    assert names == AdanosMarketSentimentToolSpec.spec_functions


def test_adanos_tool_requires_credentials_and_valid_timeout():
    with pytest.raises(ValueError, match="API key"):
        AdanosMarketSentimentToolSpec(" ")
    with pytest.raises(ValueError, match="timeout"):
        AdanosMarketSentimentToolSpec(TEST_VALUE, timeout=0)


def test_get_stock_sentiment_uses_source_and_date_window(adanos_tool, mocker):
    response = mock_response(mocker, {"ticker": "AAPL", "buzz_score": 72.0})
    get = mocker.patch.object(adanos_tool._session, "get", return_value=response)

    result = adanos_tool.get_adanos_stock_sentiment(
        "aapl", source="news", start_date="2026-08-01", end_date="2026-08-07"
    )

    assert result == {"ticker": "AAPL", "buzz_score": 72.0}
    get.assert_called_once_with(
        "https://api.adanos.org/news/stocks/v1/stock/AAPL",
        params={"from": "2026-08-01", "to": "2026-08-07"},
        timeout=30.0,
    )
    response.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize("ticker", ["AAPL/%6d%65%6e%74%69%6f%6e%73", "AAPL/mentions"])
def test_get_stock_sentiment_rejects_path_injection(adanos_tool, ticker):
    with pytest.raises(ValueError, match="valid stock ticker"):
        adanos_tool.get_adanos_stock_sentiment(ticker)


def test_get_crypto_sentiment_uses_reddit_crypto_endpoint(adanos_tool, mocker):
    response = mock_response(mocker, {"symbol": "BTC"})
    get = mocker.patch.object(adanos_tool._session, "get", return_value=response)

    assert adanos_tool.get_adanos_crypto_sentiment("btc") == {"symbol": "BTC"}
    get.assert_called_once_with(
        "https://api.adanos.org/reddit/crypto/v1/token/BTC",
        params={},
        timeout=30.0,
    )

    with pytest.raises(ValueError, match="valid crypto symbol"):
        adanos_tool.get_adanos_crypto_sentiment("BTC/%2fmentions")


def test_trending_assets_validates_limit_and_crypto_source(adanos_tool, mocker):
    response = mock_response(mocker, [{"symbol": "BTC", "buzz_score": 80.0}])
    get = mocker.patch.object(adanos_tool._session, "get", return_value=response)

    assert adanos_tool.get_adanos_trending_assets(asset_type="crypto", limit=5) == [
        {"symbol": "BTC", "buzz_score": 80.0}
    ]
    get.assert_called_once_with(
        "https://api.adanos.org/reddit/crypto/v1/trending",
        params={"limit": 5},
        timeout=30.0,
    )

    with pytest.raises(ValueError, match="between 1 and 100"):
        adanos_tool.get_adanos_trending_assets(limit=0)
    with pytest.raises(ValueError, match="reddit only"):
        adanos_tool.get_adanos_trending_assets(asset_type="crypto", source="news")


def test_compare_assets_normalizes_symbols(adanos_tool, mocker):
    response = mock_response(mocker, {"comparison": []})
    get = mocker.patch.object(adanos_tool._session, "get", return_value=response)

    assert adanos_tool.compare_adanos_asset_sentiment(
        [" aapl ", "nvda", "AAPL"], source="x"
    ) == {"comparison": []}
    get.assert_called_once_with(
        "https://api.adanos.org/x/stocks/v1/compare",
        params={"tickers": "AAPL,NVDA"},
        timeout=30.0,
    )

    with pytest.raises(ValueError, match="at least one"):
        adanos_tool.compare_adanos_asset_sentiment([])
    with pytest.raises(ValueError, match="valid stock ticker"):
        adanos_tool.compare_adanos_asset_sentiment(["AAPL,NVDA"])
    with pytest.raises(ValueError, match="at most 10"):
        adanos_tool.compare_adanos_asset_sentiment([f"A{index}" for index in range(11)])


def test_market_sentiment_rejects_unknown_values(adanos_tool):
    with pytest.raises(ValueError, match="source must be one of"):
        adanos_tool.get_adanos_market_sentiment(source="stocktwits")
    with pytest.raises(ValueError, match="asset_type"):
        adanos_tool.get_adanos_market_sentiment(asset_type="forex")


def test_adanos_http_errors_are_not_swallowed(adanos_tool, mocker):
    response = mock_response(mocker, {})
    response.raise_for_status.side_effect = requests.HTTPError("rate limited")
    mocker.patch.object(adanos_tool._session, "get", return_value=response)

    with pytest.raises(requests.HTTPError, match="rate limited"):
        adanos_tool.get_adanos_market_sentiment()


def test_adanos_rate_limit_error_includes_retry_guidance(adanos_tool, mocker):
    response = mock_response(
        mocker,
        {"detail": {"message": "Too many requests for this account."}},
    )
    response.status_code = 429
    response.headers = {"Retry-After": "42"}
    mocker.patch.object(adanos_tool._session, "get", return_value=response)

    with pytest.raises(
        requests.HTTPError,
        match=r"Too many requests for this account\. Retry after 42 seconds\.",
    ) as exc_info:
        adanos_tool.get_adanos_market_sentiment()

    assert exc_info.value.response is response
    response.raise_for_status.assert_not_called()
