"""Wolfram Alpha tool spec."""

import urllib.parse
from typing import Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

QUERY_URL_TMPL = "http://api.wolframalpha.com/v1/result?appid={app_id}&i={query}"


class WolframAlphaToolSpec(BaseToolSpec):
    """Wolfram Alpha tool spec."""

    spec_functions = ["wolfram_alpha_query"]

    def __init__(self, app_id: Optional[str] = None) -> None:
        """Initialize with parameters."""
        self.token = app_id

    def wolfram_alpha_query(self, query: str):
        """
        Make a query to wolfram alpha about a mathematical or scientific problem.

        Example inputs:
            "(7 * 12 ^ 10) / 321"
            "How many calories are there in a pound of strawberries"

        Args:
            query (str): The query to be passed to wolfram alpha.

        """
        response = requests.get(
            QUERY_URL_TMPL.format(
                app_id=self.token, query=urllib.parse.quote_plus(query)
            )
        )
        return response.text
