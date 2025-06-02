"""
PostgresML index.
An index that is built on top of PostgresML.
"""

import logging
from typing import Any, List, Optional, Dict

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.indices.managed.postgresml.base import PostgresMLIndex

_logger = logging.getLogger(__name__)


class PostgresMLRetriever(BaseRetriever):
    """
    PostgresML Retriever.

    Args:
        index (PostgresMLIndex): the PostgresML Index

    """

    def __init__(
        self,
        index: PostgresMLIndex,
        callback_manager: Optional[CallbackManager] = None,
        pgml_query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 5,
        rerank: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._pgml_query = pgml_query
        self._limit = limit
        self._rerank = rerank
        super().__init__(callback_manager)

    def _retrieve(
        self,
        query_bundle: Optional[QueryBundle] = None,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        return run_async_tasks([self._aretrieve(query_bundle, **kwargs)])[0]

    async def _aretrieve(
        self,
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        async def do_vector_search():
            if self._pgml_query:
                return await self._index.collection.vector_search(
                    self._pgml_query,
                    self._index.pipeline,
                )
            else:
                if not query_bundle:
                    raise Exception(
                        "Must provide either query or query_bundle to retrieve and aretrieve"
                    )
                if self._rerank is not None:
                    self._rerank = self._rerank | {"query": query_bundle.query_str}
                return await self._index.collection.vector_search(
                    {
                        "query": {
                            "fields": {
                                "content": {
                                    "query": query_bundle.query_str,
                                    "parameters": {"prompt": "query: "},
                                }
                            }
                        },
                        "rerank": self._rerank,
                        "limit": self._limit,
                    },
                    self._index.pipeline,
                )

        results = await do_vector_search()
        return [
            NodeWithScore(
                node=TextNode(
                    id_=r["document"]["id"],
                    text=r["chunk"],
                    metadata=r["document"]["metadata"],
                ),
                score=r["score"],
            )
            if self._rerank is None
            else NodeWithScore(
                node=TextNode(
                    id_=r["document"]["id"],
                    text=r["chunk"],
                    metadata=r["document"]["metadata"],
                ),
                score=r["rerank_score"],
            )
            for r in results
        ]
