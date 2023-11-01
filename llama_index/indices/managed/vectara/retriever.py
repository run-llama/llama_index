"""Vectara index.
An index that that is built on top of Vectara.
"""

import json
import logging
from typing import Any, List

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.managed.vectara.base import VectaraIndex
from llama_index.indices.query.schema import QueryBundle
from llama_index.schema import NodeWithScore, TextNode

_logger = logging.getLogger(__name__)


class VectaraRetriever(BaseRetriever):
    """Vectara Retriever.

    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return.
        lambda_val (float): for hybrid search.
            0 = neural search only.
            1 = keyword match only.
            In between values are a linear interpolation
        n_sentences_before (int):
            number of sentences before the matched sentence to return in the node
        n_sentences_after (int):
             number of sentences after the matched sentence to return in the node
        filter: metadata filter (if specified)
    """

    def __init__(
        self,
        index: VectaraIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        lambda_val: float = 0.025,
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        filter: str = "",
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
            "X-Source": "llama_index",
        }

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Query Vectara index to get for top k most similar nodes.

        Args:
            query: Query Bundle
        """
        self._similarity_top_k
        corpus_key = {
            "customer_id": self._index._vectara_customer_id,
            "corpus_id": self._index._vectara_corpus_id,
            "lexical_interpolation_config": {"lambda": self._lambda_val},
        }
        if len(self._filter) > 0:
            corpus_key["metadataFilter"] = self._filter

        data = json.dumps(
            {
                "query": [
                    {
                        "query": query_bundle.query_str,
                        "start": 0,
                        "num_results": self._similarity_top_k,
                        "context_config": {
                            "sentences_before": self._n_sentences_before,
                            "sentences_after": self._n_sentences_after,
                        },
                        "corpus_key": [corpus_key],
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
            return []

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
