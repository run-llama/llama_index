"""Vectara index.
An index that that is built on top of Vectara
"""

import logging
from typing import Any, List, Optional, Sequence, Dict
import json

from llama_index.schema import TextNode
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore
from llama_index.indices.managed.base import ManagedIndex
from dataclasses import dataclass
from llama_index.schema import BaseNode
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K

_logger = logging.getLogger(__name__)


@dataclass
class VectaraQuery:
    """Vectara query."""

    query_str: Optional[str] = None
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K

    # For Hybrid search: 0 = neural search only, 1 = keyword match only. In between values are a linear interpolation
    # see https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching for more details
    lambda_val: Optional[float] = None

    # define how many sentences before/after the matched sentence to return in the node
    n_sentences_before: int = 2
    n_sentences_after: int = 2

    # metadata filters
    filter: Optional[str] = None


@dataclass
class VectaraQueryResult:
    """Vectara query result."""

    nodes: Optional[Sequence[BaseNode]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None


class VectaraRetriever(BaseRetriever):
    """Vectara Retriever.
    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return.
        lambda_val (float): for hybrid search, 0 = neural search only, 1 = keyword match only. In between values are a linear interpolation
        n_sentences_before (int): number of sentences before the matched sentence to return in the node
        n_sentences_after (int): number of sentences after the matched sentence to return in the node
        filter: metadata filter (if specified)
    """

    def __init__(
        self,
        index: ManagedIndex,
        similarity_top_k: Optional[int] = DEFAULT_SIMILARITY_TOP_K,
        lambda_val: Optional[float] = None,
        n_sentences_before: Optional[int] = 2,
        n_sentences_after: Optional[int] = 2,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._lambda_val = lambda_val
        self._n_sentences_before = n_sentences_before
        self._n_sentences_after = n_sentences_after
        self._filter = filter

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._index._vectara_api_key,
            "customer-id": self._index._vectara_customer_id,
            "Content-Type": "application/json",
        }

    def _retrieve(
        self,
        query: VectaraQuery,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Query Vectara index to get for top k most similar nodes.

        Args:
            query: VectaraQuery which includes the query_str, similarity_top_k, alpha (for hybrid search) and filters
        """

        similarity_top_k = (
            query.similarity_top_k
            if query.similarity_top_k
            else self._similarity_top_k
            if self._similarity_top_k
            else DEFAULT_SIMILARITY_TOP_K
        )
        n_sentences_before = (
            query.n_sentences_before
            if query.n_sentences_before
            else self._n_sentences_before
            if self._n_sentences_before
            else 2
        )
        n_sentences_after = (
            query.n_sentences_after
            if query.n_sentences_after
            else self._n_sentences_after
            if self._n_sentences_after
            else 2
        )
        lambda_val = (
            query.lambda_val
            if query.lambda_val
            else self._lambda_val
            if self._lambda_val
            else 0.025
        )
        data = json.dumps(
            {
                "query": [
                    {
                        "query": query.query_str,
                        "start": 0,
                        "num_results": similarity_top_k,
                        "context_config": {
                            "sentences_before": n_sentences_before,
                            "sentences_after": n_sentences_after,
                        },
                        "corpus_key": [
                            {
                                "customer_id": self._index._vectara_customer_id,
                                "corpus_id": self._index._vectara_corpus_id,
                                "metadataFilter": query.filter,
                                "lexical_interpolation_config": {"lambda": lambda_val},
                            }
                        ],
                    }
                ]
            }
        )

        response = self._index._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=data,
            timeout=self._index.vectara_api_timeout,
        )

        if response.status_code != 200:
            _logger.error(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return VectaraQueryResult(nodes=[], similarities=[], ids=[])

        result = response.json()
        responses = result["responseSet"][0]["response"]
        documents = result["responseSet"][0]["document"]

        metadatas = []
        for x in responses:
            md = {m["name"]: m["value"] for m in x["metadata"]}
            doc_num = x["documentIndex"]
            doc_md = {m["name"]: m["value"] for m in documents[doc_num]["metadata"]}
            md.update(doc_md)
            metadatas.append(md)

        top_k_nodes = []
        for x, md in zip(responses, metadatas):
            doc_inx = x["documentIndex"]
            doc_id = documents[doc_inx]["id"]
            node = NodeWithScore(
                node=TextNode(text=x["text"], id_=doc_id, metadata=md), score=x["score"]
            )
            top_k_nodes.append(node)

        return top_k_nodes
