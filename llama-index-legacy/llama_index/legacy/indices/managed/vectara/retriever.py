"""Vectara index.
An index that is built on top of Vectara.
"""

import json
import logging
from typing import Any, List, Optional, Tuple

from llama_index.legacy.callbacks.base import CallbackManager
from llama_index.legacy.core.base_retriever import BaseRetriever
from llama_index.legacy.indices.managed.types import ManagedIndexQueryMode
from llama_index.legacy.indices.managed.vectara.base import VectaraIndex
from llama_index.legacy.indices.managed.vectara.prompts import (
    DEFAULT_VECTARA_QUERY_PROMPT_TMPL,
)
from llama_index.legacy.indices.vector_store.retrievers.auto_retriever.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.legacy.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.legacy.vector_stores.types import (
    FilterCondition,
    MetadataFilters,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)

_logger = logging.getLogger(__name__)


class VectaraRetriever(BaseRetriever):
    """Vectara Retriever.

    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return, defaults to 5.
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
        mmr_k: number of results to fetch for MMR, defaults to 50
        mmr_diversity_bias: number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding
            to minimum diversity and 1 to maximum diversity.
            Defaults to 0.3.
        summary_enabled: whether to generate summaries or not. Defaults to False.
        summary_response_lang: language to use for summary generation.
        summary_num_results: number of results to use for summary generation.
        summary_prompt_name: name of the prompt to use for summary generation.
    """

    def __init__(
        self,
        index: VectaraIndex,
        similarity_top_k: int = 5,
        vectara_query_mode: ManagedIndexQueryMode = ManagedIndexQueryMode.DEFAULT,
        lambda_val: float = 0.025,
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        filter: str = "",
        mmr_k: int = 50,
        mmr_diversity_bias: float = 0.3,
        summary_enabled: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 7,
        summary_prompt_name: str = "vectara-experimental-summary-ext-2023-10-23-small",
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._lambda_val = lambda_val
        self._n_sentences_before = n_sentences_before
        self._n_sentences_after = n_sentences_after
        self._filter = filter

        if vectara_query_mode == ManagedIndexQueryMode.MMR:
            self._mmr = True
            self._mmr_k = mmr_k
            self._mmr_diversity_bias = mmr_diversity_bias
        else:
            self._mmr = False

        if summary_enabled:
            self._summary_enabled = True
            self._summary_response_lang = summary_response_lang
            self._summary_num_results = summary_num_results
            self._summary_prompt_name = summary_prompt_name
        else:
            self._summary_enabled = False
        super().__init__(callback_manager)

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
        """Retrieve top k most similar nodes.

        Args:
            query: Query Bundle
        """
        return self._vectara_query(query_bundle, **kwargs)[0]  # return top_nodes only

    def _vectara_query(
        self,
        query_bundle: QueryBundle,
        **kwargs: Any,
    ) -> Tuple[List[NodeWithScore], str]:
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

        if self._summary_enabled:
            data["query"][0]["summary"] = [
                {
                    "responseLang": self._summary_response_lang,
                    "maxSummarizedResults": self._summary_num_results,
                    "summarizerPromptName": self._summary_prompt_name,
                }
            ]

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
            return [], ""

        result = response.json()

        responses = result["responseSet"][0]["response"]
        documents = result["responseSet"][0]["document"]
        summary = (
            result["responseSet"][0]["summary"][0]["text"]
            if self._summary_enabled
            else None
        )

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
                node=TextNode(text=x["text"], id_=doc_id, metadata=md), score=x["score"]  # type: ignore
            )
            top_nodes.append(node)

        return top_nodes[: self._similarity_top_k], summary

    async def _avectara_query(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], str]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._vectara_query(query_bundle)


class VectaraAutoRetriever(VectorIndexAutoRetriever):
    """Managed Index auto retriever.

    A retriever for a Vectara index that uses an LLM to automatically set
    filtering query parameters.
    Based on VectorStoreAutoRetriever, and uses some of the vector_store
    types that are associated with auto retrieval.

    Args:
        index (VectaraIndex): Vectara Index instance
        vector_store_info (VectorStoreInfo): additional information about
            vector store content and supported metadata filters. The natural language
            description is used by an LLM to automatically set vector store query
            parameters.
        Other variables are the same as VectorStoreAutoRetriever or VectaraRetriever
    """

    def __init__(
        self,
        index: VectaraIndex,
        vector_store_info: VectorStoreInfo,
        **kwargs: Any,
    ) -> None:
        super().__init__(index, vector_store_info, prompt_template_str=DEFAULT_VECTARA_QUERY_PROMPT_TMPL, **kwargs)  # type: ignore
        self._index = index  # type: ignore
        self._kwargs = kwargs
        self._verbose = self._kwargs.get("verbose", False)
        self._explicit_filter = self._kwargs.pop("filter", "")

    def _build_retriever_from_spec(
        self, spec: VectorStoreQuerySpec
    ) -> Tuple[VectaraRetriever, QueryBundle]:
        query_bundle = self._get_query_bundle(spec.query)

        filter_list = [
            (filter.key, filter.operator.value, filter.value) for filter in spec.filters
        ]
        if self._verbose:
            print(f"Using query str: {spec.query}")
            print(f"Using implicit filters: {filter_list}")

        # create filter string from implicit filters
        if len(spec.filters) == 0:
            filter_str = ""
        else:
            filters = MetadataFilters(
                filters=[*spec.filters, *self._extra_filters.filters]
            )
            condition = " and " if filters.condition == FilterCondition.AND else " or "
            filter_str = condition.join(
                [
                    f"(doc.{f.key} {f.operator.value} '{f.value}')"
                    for f in filters.filters
                ]
            )

        # add explicit filter if specified
        if self._explicit_filter:
            if len(filter_str) > 0:
                filter_str = f"({filter_str}) and ({self._explicit_filter})"
            else:
                filter_str = self._explicit_filter

        if self._verbose:
            print(f"final filter string: {filter_str}")

        return (
            VectaraRetriever(
                index=self._index,  # type: ignore
                filter=filter_str,
                **self._kwargs,
            ),
            query_bundle,
        )

    def _vectara_query(
        self,
        query_bundle: QueryBundle,
        **kwargs: Any,
    ) -> Tuple[List[NodeWithScore], str]:
        spec = self.generate_retrieval_spec(query_bundle)
        vectara_retriever, new_query = self._build_retriever_from_spec(
            VectorStoreQuerySpec(
                query=spec.query, filters=spec.filters, top_k=self._similarity_top_k
            )
        )
        return vectara_retriever._vectara_query(new_query, **kwargs)
