"""Wolfram Alpha tool spec."""

import urllib.parse
from typing import Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

QUERY_URL_TMPL = "https://www.wolframalpha.com/api/v1/llm-api"


class WolframAlphaToolSpec(BaseToolSpec):
    """Wolfram Alpha tool spec."""

    spec_functions = ["wolfram_alpha_query"]

    def __init__(
        self,
        app_id: Optional[str] = None,
        api_params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        self.token = app_id
        self._api_params = api_params or {}

    def wolfram_alpha_query(self, query: str) -> str:
        r"""
        Query Wolfram|Alpha for computational knowledge.

        WolframAlpha understands natural language queries about entities in
        chemistry, physics, geography, history, art, astronomy, and more.
        WolframAlpha performs mathematical calculations, date and unit
        conversions, formula solving, etc.

        QUERY FORMAT:
        - Convert inputs to simplified keyword queries whenever possible
          (e.g., "how many people live in France" -> "France population")
        - Send queries in English only; translate non-English queries before
          sending, then respond in the original language
        - Query must be a single-line string

        MATH AND UNITS:
        - ALWAYS use this exponent notation: 6*10^14, NEVER 6e14
        - Use ONLY single-letter variable names, with or without integer
          subscript (e.g., n, n1, n_1)
        - Use named physical constants (e.g., "speed of light") without
          numerical substitution
        - Include a space between compound units (e.g., "Ω m" for "ohm*meter")
        - To solve for a variable in an equation with units, consider solving
          a corresponding equation without units; exclude counting units
          (e.g., books), include genuine units (e.g., kg)

        MULTIPLE PROPERTIES:
        - If data for multiple properties is needed, make separate calls for
          each property

        HANDLING RESULTS:
        - Display image URLs with Markdown syntax: ![URL]
        - Use proper Markdown formatting for all math, scientific, and chemical
          formulas, symbols, etc.: '$$\n[expression]\n$$' for standalone
          cases and '\( [expression] \)' when inline
        - Never mention your knowledge cutoff date; Wolfram may return more
          recent data
        - If a WolframAlpha result is not relevant to the query:
          - If Wolfram provides multiple 'Assumptions' for a query, choose the
            more relevant one(s) without explaining the initial result
          - Re-send the exact same 'input' with NO modifications, and add the
            'assumption' parameter with the relevant values
          - ONLY simplify or rephrase the initial query if a more relevant
            'Assumption' or other input suggestions are not provided
          - Do not explain each step unless user input is needed; proceed
            directly to making a better API call based on available assumptions

        Args:
            query (str): The query to be passed to wolfram alpha.

        """
        params = {"input": query, **self._api_params}
        url = f"{QUERY_URL_TMPL}?{urllib.parse.urlencode(params)}"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
