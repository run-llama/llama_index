"""You Retriever."""

import logging
import os
from typing import List, Optional

import requests

from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.indices.query.schema import QueryBundle
from llama_index.legacy.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


class YouRetriever(BaseRetriever):
    """You retriever."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self._api_key = api_key or os.environ["YOU_API_KEY"]
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        headers = {"X-API-Key": self._api_key}
        results = requests.get(
            f"https://api.ydc-index.io/search?query={query_bundle.query_str}",
            headers=headers,
        ).json()

        search_hits = ["\n".join(hit["snippets"]) for hit in results["hits"]]
        return [NodeWithScore(node=TextNode(text=s), score=1.0) for s in search_hits]
