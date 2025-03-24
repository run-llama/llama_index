"""Dappier Real Time Search tool spec."""

import os
from typing import Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class DappierRealTimeSearchToolSpec(BaseToolSpec):
    """Dappier Real Time Search tool spec."""

    spec_functions = ["search_real_time_data", "search_stock_market_data"]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the Dappier Real Time Search tool spec.

        To obtain an API key, visit: https://platform.dappier.com/profile/api-keys
        """
        from dappier import Dappier

        self.api_key = api_key or os.environ.get("DAPPIER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set DAPPIER_API_KEY in environment variables.\n"
                "To obtain an API key, visit: https://platform.dappier.com/profile/api-keys"
            )

        self.client = Dappier(api_key=self.api_key)

    def search_real_time_data(self, query: str) -> str:
        """
        Performs a real-time data search.

        Args:
            query (str): The user-provided input string for retrieving
            real-time google web search results including the latest news,
            weather, travel, deals and more.

        Returns:
            str: A response message containing the real-time data results.
        """
        ai_model_id = "am_01j0rzq4tvfscrgzwac7jv1p4c"
        response = self.client.search_real_time_data(
            query=query, ai_model_id=ai_model_id
        )
        return response.message if response else "No real-time data found."

    def search_stock_market_data(self, query: str) -> str:
        """
        Performs a stock market data search.

        Args:
            query (str): The user-provided input string for retrieving
            real-time financial news, stock prices, and trades from polygon.io,
            with AI-powered insights and up-to-the-minute updates to keep you
            informed on all your financial interests.

        Returns:
            str: A response message containing the stock market data results.
        """
        ai_model_id = "am_01j749h8pbf7ns8r1bq9s2evrh"
        response = self.client.search_real_time_data(
            query=query, ai_model_id=ai_model_id
        )
        return response.message if response else "No stock market data found."
