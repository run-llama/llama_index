"""FXMacroData tool spec."""

import os
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.fxmacrodata.com/v1"
DEFAULT_TIMEOUT = 30


class FXMacroDataToolSpec(BaseToolSpec):
    """
    FXMacroData tool spec.

    Official-source macroeconomic, FX and central-bank data for 18 currencies.
    FXMacroData aggregates official publishers - statistical agencies, central
    banks and exchanges - behind one contract, so an agent does not need to know
    which of eighteen publishers to call or how each one formats its data. Every
    observation carries the instant it was published, so a value is never
    presented as though it were known before its release.

    USD works without an API key. A key widens the history window and unlocks the
    other seventeen currencies plus FX rates, rate differentials, COT positioning
    and commodities.
    """

    spec_functions = [
        "search_indicators",
        "get_latest_macro_snapshot",
        "get_indicator_history",
        "get_release_calendar",
        "get_central_bank_headlines",
        "get_fx_rate",
        "get_rate_differential",
        "get_cot_positioning",
        "get_commodity_prices",
        "get_market_sessions",
        "get_risk_sentiment",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the FXMacroData tool spec.

        Args:
            api_key (Optional[str]): FXMacroData API key. Optional, since USD data
                is public. Falls back to the FXMACRODATA_API_KEY environment variable.
            base_url (str): API base URL. Override only to target a different deployment.
            timeout (int): Per-request HTTP timeout in seconds.

        """
        self.api_key = api_key or os.getenv("FXMACRODATA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Call the FXMacroData API and return the raw JSON body."""
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if clean:
            url = f"{url}?{urllib.parse.urlencode(clean)}"

        # base_url is settable, so pin the scheme: urlopen would otherwise honour
        # file:// and read from the local filesystem.
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            return "FXMacroData refused a request over an unsupported URL scheme."

        headers = {"Accept": "application/json"}
        if self.api_key:
            # Header rather than a query parameter, so the key stays out of proxy
            # and server access logs.
            headers["X-API-Key"] = self.api_key

        request = urllib.request.Request(url, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return (
                    f"FXMacroData denied the request to {endpoint} (HTTP {exc.code}). "
                    "This data requires an API key; USD macro data is available without one."
                )
            return f"FXMacroData request to {endpoint} failed with HTTP {exc.code}."
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return f"FXMacroData request to {endpoint} failed: {exc}"

    def search_indicators(self, currency: str = "USD") -> str:
        """
        List every macroeconomic indicator published for a currency.

        Call this first when the indicator slug is not already known. The response
        gives the slug to pass to get_indicator_history, with its unit, frequency,
        publisher and coverage.

        Args:
            currency (str): Three-letter currency code, for example 'USD' or 'EUR'.

        """
        return self._request(f"data_catalogue/{currency.lower()}")

    def get_latest_macro_snapshot(self, currency: str = "USD") -> str:
        """
        Return the latest value of every indicator for a currency in one call.

        This is the fastest way to read the current macro picture for an economy:
        one request returns the most recent print for each indicator with its
        publication timestamp, previous value and change.

        Args:
            currency (str): Three-letter currency code, for example 'USD' or 'JPY'.

        """
        return self._request(f"announcements/{currency.lower()}/latest")

    def get_indicator_history(
        self,
        currency: str = "USD",
        indicator: str = "inflation",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """
        Return the published history of one macroeconomic indicator.

        Each row carries the value, the period it covers and the instant it was
        announced.

        Args:
            currency (str): Three-letter currency code, for example 'USD'.
            indicator (str): Indicator slug from search_indicators, for example
                'inflation', 'non_farm_payrolls' or 'policy_rate'.
            start_date (Optional[str]): Optional ISO start date, e.g. '2024-01-01'.
            end_date (Optional[str]): Optional ISO end date.
            limit (int): Maximum rows to return. The API caps this at 100.

        """
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        }
        return self._request(f"announcements/{currency.lower()}/{indicator}", params)

    def get_release_calendar(self, currency: str = "USD", limit: int = 20) -> str:
        """
        Return upcoming scheduled macroeconomic releases for a currency.

        Use this to know what is due and when, before it happens.

        Args:
            currency (str): Three-letter currency code, for example 'USD'.
            limit (int): Maximum releases to return. The API caps this at 100.

        """
        return self._request(f"calendar/{currency.lower()}", {"limit": limit})

    def get_central_bank_headlines(self, currency: str = "USD", limit: int = 10) -> str:
        """
        Return recent official central-bank press releases for a currency.

        Args:
            currency (str): Three-letter currency code, for example 'USD' or 'EUR'.
            limit (int): Maximum headlines to return.

        """
        return self._request(f"press-releases/{currency.lower()}", {"limit": limit})

    def get_fx_rate(
        self, base: str = "EUR", quote: str = "USD", limit: int = 10
    ) -> str:
        """
        Return official reference exchange rates for a currency pair.

        Rates come from official publishers such as the ECB and the Federal
        Reserve rather than a broker feed. Requires an API key.

        Args:
            base (str): Three-letter base currency code, for example 'EUR'.
            quote (str): Three-letter quote currency code, for example 'USD'.
            limit (int): Maximum observations to return, newest first.

        """
        return self._request(f"forex/{base.lower()}/{quote.lower()}", {"limit": limit})

    def get_rate_differential(
        self, base: str = "USD", quote: str = "JPY", limit: int = 10
    ) -> str:
        """
        Return the policy rate differential between two currencies.

        The rate differential is the standard first look at carry for a pair.
        Requires an API key.

        Args:
            base (str): Three-letter base currency code, for example 'USD'.
            quote (str): Three-letter quote currency code, for example 'JPY'.
            limit (int): Maximum observations to return, newest first.

        """
        return self._request(
            f"rate_differentials/{base.lower()}/{quote.lower()}", {"limit": limit}
        )

    def get_cot_positioning(self, currency: str = "USD", limit: int = 10) -> str:
        """
        Return CFTC Commitment of Traders positioning for a currency.

        Shows how speculative and commercial participants are positioned, the
        usual proxy for crowding in a currency. Requires an API key.

        Args:
            currency (str): Three-letter currency code, for example 'GBP'.
            limit (int): Maximum weekly reports to return, newest first.

        """
        return self._request(f"cot/{currency.lower()}", {"limit": limit})

    def get_commodity_prices(self) -> str:
        """
        Return the latest official prices for tracked commodities.

        Requires an API key.
        """
        return self._request("commodities/latest")

    def get_market_sessions(self) -> str:
        """
        Return the current FX market session status.

        Tells you which of the Sydney, Tokyo, London and New York sessions are
        open now, which governs when liquidity is available.
        """
        return self._request("market_sessions")

    def get_risk_sentiment(self) -> str:
        """Return the current cross-asset risk sentiment reading."""
        return self._request("risk_sentiment")


__all__: List[str] = ["FXMacroDataToolSpec"]
