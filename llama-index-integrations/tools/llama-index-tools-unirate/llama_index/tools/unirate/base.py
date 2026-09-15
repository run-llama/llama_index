"""UniRate tool spec — currency exchange rates, conversion, and VAT for LlamaIndex agents."""

from typing import Any, Dict, List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class UnirateToolSpec(BaseToolSpec):
    """
    UniRate tool spec.

    Gives an agent real-time currency exchange rates, currency conversion, the list
    of supported currencies, and VAT rates via the UniRate API
    (https://unirateapi.com). A free API key is required.
    """

    spec_functions = [
        "get_exchange_rate",
        "convert_currency",
        "list_supported_currencies",
        "get_vat_rates",
    ]

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        """
        Initialize with a UniRate API key.

        Args:
            api_key (str): Your UniRate API key (free tier available at https://unirateapi.com).
            timeout (int): Request timeout in seconds. Defaults to 30.

        """
        from unirate import UnirateClient

        self.client = UnirateClient(api_key=api_key, timeout=timeout)

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get the current exchange rate between two currencies.

        Args:
            from_currency (str): The source currency code, e.g. "USD".
            to_currency (str): The target currency code, e.g. "EUR".

        Returns:
            float: How many units of ``to_currency`` equal one unit of ``from_currency``.

        """
        return self.client.get_rate(from_currency=from_currency, to_currency=to_currency)

    def convert_currency(
        self, amount: float, from_currency: str, to_currency: str
    ) -> float:
        """
        Convert an amount of money from one currency to another at the current rate.

        Args:
            amount (float): The amount to convert.
            from_currency (str): The source currency code, e.g. "USD".
            to_currency (str): The target currency code, e.g. "GBP".

        Returns:
            float: The converted amount expressed in ``to_currency``.

        """
        return self.client.convert(
            to_currency=to_currency, amount=amount, from_currency=from_currency
        )

    def list_supported_currencies(self) -> List[str]:
        """
        List every currency code supported by the UniRate API.

        Returns:
            List[str]: Supported ISO currency codes (and crypto symbols).

        """
        return self.client.get_supported_currencies()

    def get_vat_rates(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get value-added-tax (VAT) rates for all countries or a single country.

        Args:
            country (Optional[str]): A two-letter country code, e.g. "DE". If omitted,
                VAT rates for all available countries are returned.

        Returns:
            dict: VAT-rate information.

        """
        return self.client.get_vat_rates(country=country)
