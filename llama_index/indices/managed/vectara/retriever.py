"""Vectara index.
An index that that is built on top of Vectara.
"""

import json
import logging
from typing import Any, Dict, List

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core import BaseRetriever
from llama_index.indices.managed.types import ManagedIndexQueryMode
from llama_index.indices.managed.vectara.base import VectaraIndex
from llama_index.schema import NodeWithScore, QueryBundle, TextNode

_logger = logging.getLogger(__name__)


class VectaraRetriever(BaseRetriever):
    """Vectara Retriever.

    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return.
        vectara_query_mode (str): vector store query mode
            See reference for vectara_query_mode for full list of supported modes.
        lambda_val (float): for hybrid search.
            0 = neural search only.
            1 = keyword match only.
            In between values are a linear interpolation
        n_sentences_before (int):
            number of sentences before the matched sentence to return in the node
        n_sentences_after (int):
             number of sentences after the matched sentence to return in the node
        filter: metadata filter (if specified)
        vectara_kwargs (dict): Additional vectara specific kwargs to pass
            through to Vectara at query time.
            * mmr_k: number of results to fetch for MMR, defaults to 50
            * mmr_diversity_bias: number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to minimum diversity and 1 to maximum diversity.
                Defaults to 0.3.
    """

    def __init__(
        self,
        index: VectaraIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vectara_query_mode: ManagedIndexQueryMode = ManagedIndexQueryMode.DEFAULT,
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
        self._kwargs: Dict[str, Any] = kwargs.get("vectara_kwargs", {})

        if vectara_query_mode == ManagedIndexQueryMode.MMR:
            self._mmr = True
            self._mmr_k = kwargs.get("mmr_k", 50)
            self._mmr_diversity_bias = kwargs.get("mmr_diversity_bias", 0.3)
        else:
            self._mmr = False

    def _get_post_headers(self) -> dict:
        """Returns headers that should be attached to each post request."""
        return {
            "x-api-key": self._index._vectara_api_key,
            "customer-id": self._index._vectara_customer_id,
            "Content-Type": "application/json",
            "X-Source": "llama_index",
        }

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
        **kwargs: Any,
    ) -> List[NodeWithScore]:
        """Query Vectara index to get for top k most similar nodes.

        Args:
            query: Query Bundle
        """
        corpus_key = {
            "customerId": self._index._vectara_customer_id,
            "corpusId": self._index._vectara_corpus_id,
            "lexicalInterpolationConfig": {"lambda": self._lambda_val},
        }
        if len(self._filter) > 0:
            corpus_key["metadataFilter"] = self._filter

        data = {
            "query": [
                {
                    "query": query_bundle.query_str,
                    "start": 0,
                    "numResults": self._mmr_k if self._mmr else self._similarity_top_k,
                    "contextConfig": {
                        "sentencesBefore": self._n_sentences_before,
                        "sentencesAfter": self._n_sentences_after,
                    },
                    "corpusKey": [corpus_key],
                }
            ]
        }
        if self._mmr:
            data["query"][0]["rerankingConfig"] = {
                "rerankerId": 272725718,
                "mmrConfig": {"diversityBias": self._mmr_diversity_bias},
            }

        response = self._index._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/query",
            data=json.dumps(data),
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

        top_nodes = []
        for x, md in zip(responses, metadatas):
            doc_inx = x["documentIndex"]
            doc_id = documents[doc_inx]["id"]
            node = NodeWithScore(
                node=TextNode(text=x["text"], id_=doc_id, metadata=md), score=x["score"]
            )
            top_nodes.append(node)

        return top_nodes[: self._similarity_top_k]
