"""Adanos market sentiment tool spec."""

import re
from typing import Any, Dict, List, Optional, Union, cast

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class AdanosMarketSentimentToolSpec(BaseToolSpec):
    """Expose Adanos stock and crypto market sentiment data to agents."""

    spec_functions = [
        "get_adanos_stock_sentiment",
        "get_adanos_crypto_sentiment",
        "get_adanos_trending_assets",
        "compare_adanos_asset_sentiment",
        "get_adanos_market_sentiment",
    ]

    _STOCK_SOURCE_PREFIXES = {
        "reddit": "/reddit/stocks/v1",
        "x": "/x/stocks/v1",
        "news": "/news/stocks/v1",
        "polymarket": "/polymarket/stocks/v1",
    }
    _CRYPTO_PREFIX = "/reddit/crypto/v1"
    _STOCK_TICKER_PATTERN = re.compile(
        r"^\$?(?:[A-Za-z0-9]{1,8}[.-][A-Za-z]|"
        r"(?=[A-Za-z0-9]{1,10}$)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]+|"
        r"[0-9]{3,10})$"
    )
    _CRYPTO_SYMBOL_PATTERN = re.compile(r"^\$?[A-Za-z0-9]{1,20}$")

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.adanos.org",
        timeout: float = 30.0,
    ) -> None:
        """Initialize the tool with an Adanos API key."""
        if not api_key.strip():
            raise ValueError("An Adanos API key is required.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "X-API-Key": api_key,
            }
        )

    @staticmethod
    def _date_params(
        start_date: Optional[str], end_date: Optional[str]
    ) -> Dict[str, str]:
        params = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date
        return params

    def _stock_prefix(self, source: str) -> str:
        normalized_source = source.strip().lower()
        try:
            return self._STOCK_SOURCE_PREFIXES[normalized_source]
        except KeyError as exc:
            supported = ", ".join(self._STOCK_SOURCE_PREFIXES)
            raise ValueError(f"source must be one of: {supported}.") from exc

    def _asset_prefix(self, asset_type: str, source: str) -> str:
        normalized_asset_type = asset_type.strip().lower()
        if normalized_asset_type == "stock":
            return self._stock_prefix(source)
        if normalized_asset_type == "crypto":
            if source.strip().lower() != "reddit":
                raise ValueError("Crypto sentiment is available from reddit only.")
            return self._CRYPTO_PREFIX
        raise ValueError("asset_type must be either 'stock' or 'crypto'.")

    @classmethod
    def _stock_ticker(cls, ticker: str) -> str:
        normalized_ticker = ticker.strip().upper()
        if not cls._STOCK_TICKER_PATTERN.fullmatch(normalized_ticker):
            raise ValueError("ticker is not a valid stock ticker symbol.")
        return normalized_ticker

    @classmethod
    def _crypto_symbol(cls, symbol: str) -> str:
        normalized_symbol = symbol.strip().upper()
        if not cls._CRYPTO_SYMBOL_PATTERN.fullmatch(normalized_symbol):
            raise ValueError("symbol is not a valid crypto symbol.")
        return normalized_symbol

    def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        response = self._session.get(
            f"{self._base_url}{path}", params=params, timeout=self._timeout
        )
        if response.status_code == 429:
            message = "Adanos rate limit exceeded."
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            detail = payload.get("detail", {}) if isinstance(payload, dict) else {}
            if isinstance(detail, dict) and detail.get("message"):
                message = str(detail["message"])
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                message = f"{message} Retry after {retry_after} seconds."
            raise requests.HTTPError(message, response=response)
        response.raise_for_status()
        return response.json()

    def get_adanos_stock_sentiment(
        self,
        ticker: str,
        source: str = "reddit",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get sentiment and attention metrics for one stock.

        Args:
            ticker: Stock ticker, for example ``AAPL`` or ``BRK.A``.
            source: One of ``reddit``, ``x``, ``news``, or ``polymarket``.
            start_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.
            end_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.

        """
        prefix = self._stock_prefix(source)
        return cast(
            Dict[str, Any],
            self._get(
                f"{prefix}/stock/{self._stock_ticker(ticker)}",
                self._date_params(start_date, end_date),
            ),
        )

    def get_adanos_crypto_sentiment(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Reddit sentiment and attention metrics for one crypto asset.

        Args:
            symbol: Crypto symbol, for example ``BTC`` or ``ETH``.
            start_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.
            end_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.

        """
        return cast(
            Dict[str, Any],
            self._get(
                f"{self._CRYPTO_PREFIX}/token/{self._crypto_symbol(symbol)}",
                self._date_params(start_date, end_date),
            ),
        )

    def get_adanos_trending_assets(
        self,
        asset_type: str = "stock",
        source: str = "reddit",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Rank trending stocks or crypto assets by attention.

        Args:
            asset_type: Either ``stock`` or ``crypto``.
            source: Stock source, or ``reddit`` for crypto.
            start_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.
            end_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.
            limit: Number of assets to return, from 1 to 100.

        """
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        prefix = self._asset_prefix(asset_type, source)
        params: Dict[str, Any] = self._date_params(start_date, end_date)
        params["limit"] = limit
        return cast(List[Dict[str, Any]], self._get(f"{prefix}/trending", params))

    def compare_adanos_asset_sentiment(
        self,
        symbols: List[str],
        asset_type: str = "stock",
        source: str = "reddit",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare sentiment metrics for several stocks or crypto assets.

        Args:
            symbols: One to ten stock tickers or crypto symbols to compare.
            asset_type: Either ``stock`` or ``crypto``.
            source: Stock source, or ``reddit`` for crypto.
            start_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.
            end_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.

        """
        if not symbols:
            raise ValueError("symbols must contain at least one asset.")

        prefix = self._asset_prefix(asset_type, source)
        is_crypto = asset_type.strip().lower() == "crypto"
        normalize = self._crypto_symbol if is_crypto else self._stock_ticker
        normalized_symbols = list(
            dict.fromkeys(normalize(symbol) for symbol in symbols)
        )
        if len(normalized_symbols) > 10:
            raise ValueError("symbols must contain at most 10 unique assets.")

        params = self._date_params(start_date, end_date)
        params["symbols" if is_crypto else "tickers"] = ",".join(normalized_symbols)
        return cast(Dict[str, Any], self._get(f"{prefix}/compare", params))

    def get_adanos_market_sentiment(
        self,
        asset_type: str = "stock",
        source: str = "reddit",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get an aggregate market sentiment snapshot.

        Args:
            asset_type: Either ``stock`` or ``crypto``.
            source: Stock source, or ``reddit`` for crypto.
            start_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.
            end_date: Optional inclusive UTC date in ``YYYY-MM-DD`` format.

        """
        prefix = self._asset_prefix(asset_type, source)
        return cast(
            Dict[str, Any],
            self._get(
                f"{prefix}/market-sentiment",
                self._date_params(start_date, end_date),
            ),
        )
