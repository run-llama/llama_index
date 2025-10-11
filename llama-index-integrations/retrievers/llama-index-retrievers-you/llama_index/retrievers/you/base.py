"""You Retriever."""

import logging
import os
import warnings
from typing import Any, Dict, List, Literal, Optional

import requests

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


class YouRetriever(BaseRetriever):
    """
    Retriever for You.com's Search and News API.

    [API reference](https://documentation.you.com/api-reference/)

    Args:
        api_key: you.com API key, if `YDC_API_KEY` is not set in the environment
        endpoint: you.com endpoints
        num_web_results: The max number of web results to return, must be under 20
        safesearch: Safesearch settings, one of "off", "moderate", "strict", defaults to moderate
        country: Country code, ex: 'US' for United States, see API reference for more info
        search_lang: (News API) Language codes, ex: 'en' for English, see API reference for more info
        ui_lang: (News API) User interface language for the response, ex: 'en' for English, see API reference for more info
        spellcheck: (News API) Whether to spell check query or not, defaults to True

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        endpoint: Literal["search", "news"] = "search",
        num_web_results: Optional[int] = None,
        safesearch: Optional[Literal["off", "moderate", "strict"]] = None,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        ui_lang: Optional[str] = None,
        spellcheck: Optional[bool] = None,
    ) -> None:
        """Init params."""
        # Should deprecate `YOU_API_KEY` in favour of `YDC_API_KEY` for standardization purposes
        self._api_key = api_key or os.getenv("YOU_API_KEY") or os.environ["YDC_API_KEY"]
        super().__init__(callback_manager)

        if endpoint not in ("search", "news"):
            raise ValueError('`endpoint` must be either "search" or "news"')

        # Raise warning if News API-specific fields are set but endpoint is not "news"
        if endpoint != "news":
            news_api_fields = (search_lang, ui_lang, spellcheck)
            for field in news_api_fields:
                if field:
                    warnings.warn(
                        (
                            f"News API-specific field '{field}' is set but `{endpoint=}`. "
                            "This will have no effect."
                        ),
                        UserWarning,
                    )

        self.endpoint = endpoint
        self.num_web_results = num_web_results
        self.safesearch = safesearch
        self.country = country
        self.search_lang = search_lang
        self.ui_lang = ui_lang
        self.spellcheck = spellcheck

    def _generate_params(self, query: str) -> Dict[str, Any]:
        params = {"safesearch": self.safesearch, "country": self.country}

        if self.endpoint == "search":
            params.update(
                query=query,
                num_web_results=self.num_web_results,
            )
        elif self.endpoint == "news":
            params.update(
                q=query,
                count=self.num_web_results,
                search_lang=self.search_lang,
                ui_lang=self.ui_lang,
                spellcheck=self.spellcheck,
            )

        # Remove `None` values
        return {k: v for k, v in params.items() if v is not None}

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        headers = {"X-API-Key": self._api_key}
        params = self._generate_params(query_bundle.query_str)
        response = requests.get(
            f"https://api.ydc-index.io/{self.endpoint}",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        results = response.json()

        nodes: List[TextNode] = []
        if self.endpoint == "search":
            for hit in results["hits"]:
                nodes.append(
                    TextNode(
                        text="\n".join(hit["snippets"]),
                    )
                )
        else:  # news endpoint
            for article in results["news"]["results"]:
                node = TextNode(
                    text=article["description"],
                    extra_info={"url": article["url"], "age": article["age"]},
                )
                nodes.append(node)

        return [NodeWithScore(node=node, score=1.0) for node in nodes]
