"""
Vectara index.
An index that is built on top of Vectara.
"""

import json
import logging
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum
import urllib.parse

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.vector_store.retrievers.auto_retriever.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.types import TokenGen
from llama_index.core.llms import (
    CompletionResponse,
)
from llama_index.core.vector_stores.types import (
    FilterCondition,
    MetadataFilters,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)
from llama_index.indices.managed.vectara.base import VectaraIndex
from llama_index.indices.managed.vectara.prompts import (
    DEFAULT_VECTARA_QUERY_PROMPT_TMPL,
)


_logger = logging.getLogger(__name__)

MMR_RERANKER_ID = 272725718
SLINGSHOT_RERANKER_ID = 272725719
UDF_RERANKER_ID = 272725722


class VectaraReranker(str, Enum):
    NONE = "none"
    MMR = "mmr"
    SLINGSHOT_ALT_NAME = "slingshot"
    SLINGSHOT = "multilingual_reranker_v1"
    UDF = "udf"


class VectaraRetriever(BaseRetriever):
    """
    Vectara Retriever.

    Args:
        index (VectaraIndex): the Vectara Index
        similarity_top_k (int): number of top k results to return, defaults to 5.
        reranker (str): reranker to use: none, mmr, multilingual_reranker_v1, or udf.
            Note that "multilingual_reranker_v1" is a Vectara Scale feature only.
        lambda_val (float): for hybrid search.
            0 = neural search only.
            1 = keyword match only.
            In between values are a linear interpolation
        n_sentences_before (int):
            number of sentences before the matched sentence to return in the node
        n_sentences_after (int):
            number of sentences after the matched sentence to return in the node
        filter: metadata filter (if specified)
        rerank_k: number of results to fetch for Reranking, defaults to 50.
        mmr_diversity_bias: number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding
            to minimum diversity and 1 to maximum diversity.
            Defaults to 0.3.
        udf_expression: the user defined expression for reranking results.
            See (https://docs.vectara.com/docs/learn/user-defined-function-reranker)
            for more details about syntax for udf reranker expressions.
        summary_enabled: whether to generate summaries or not. Defaults to False.
        summary_response_lang: language to use for summary generation.
        summary_num_results: number of results to use for summary generation.
        summary_prompt_name: name of the prompt to use for summary generation.
        citations_style: The style of the citations in the summary generation,
            either "numeric", "html", "markdown", or "none".
            This is a Vectara Scale only feature. Defaults to None.
        citations_url_pattern: URL pattern for html and markdown citations.
            If non-empty, specifies the URL pattern to use for citations; e.g. "{doc.url}".
            See (https://docs.vectara.com/docs/api-reference/search-apis/search
                 #citation-format-in-summary) for more details.
            This is a Vectara Scale only feature. Defaults to None.
        citations_text_pattern: The displayed text for citations.
            If not specified, numeric citations are displayed for text.
    """

    def __init__(
        self,
        index: VectaraIndex,
        similarity_top_k: int = 10,
        lambda_val: float = 0.005,
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        filter: str = "",
        reranker: VectaraReranker = VectaraReranker.NONE,
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.3,
        udf_expression: str = None,
        summary_enabled: bool = False,
        summary_response_lang: str = "eng",
        summary_num_results: int = 7,
        summary_prompt_name: str = "vectara-summary-ext-24-05-sml",
        citations_style: Optional[str] = None,
        citations_url_pattern: Optional[str] = None,
        citations_text_pattern: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        x_source_str: str = "llama_index",
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._lambda_val = lambda_val
        self._n_sentences_before = n_sentences_before
        self._n_sentences_after = n_sentences_after
        self._filter = filter
        self._citations_style = citations_style.upper() if citations_style else None
        self._citations_url_pattern = citations_url_pattern
        self._citations_text_pattern = citations_text_pattern
        self._x_source_str = x_source_str

        if reranker == VectaraReranker.MMR:
            self._rerank = True
            self._rerank_k = rerank_k
            self._mmr_diversity_bias = mmr_diversity_bias
            self._reranker_id = MMR_RERANKER_ID
        elif (
            reranker == VectaraReranker.SLINGSHOT
            or reranker == VectaraReranker.SLINGSHOT_ALT_NAME
        ):
            self._rerank = True
            self._rerank_k = rerank_k
            self._reranker_id = SLINGSHOT_RERANKER_ID
        elif reranker == VectaraReranker.UDF and udf_expression is not None:
            self._rerank = True
            self._rerank_k = rerank_k
            self._udf_expression = udf_expression
            self._reranker_id = UDF_RERANKER_ID
        else:
            self._rerank = False

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
            "X-Source": self._x_source_str,
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
        """
        Retrieve top k most similar nodes.

        Args:
            query: Query Bundle
        """
        return self._vectara_query(query_bundle, **kwargs)[0]  # return top_nodes only

    def _build_vectara_query_body(
        self,
        query_str: str,
        chat: bool = False,
        chat_conv_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict:
        corpus_keys = [
            {
                "customerId": self._index._vectara_customer_id,
                "corpusId": corpus_id,
                "lexicalInterpolationConfig": {"lambda": self._lambda_val},
            }
            for corpus_id in self._index._vectara_corpus_id.split(",")
        ]
        if len(self._filter) > 0:
            for k in corpus_keys:
                k["metadataFilter"] = self._filter

        data = {
            "query": [
                {
                    "query": query_str,
                    "start": 0,
                    "numResults": (
                        self._rerank_k if self._rerank else self._similarity_top_k
                    ),
                    "contextConfig": {
                        "sentencesBefore": self._n_sentences_before,
                        "sentencesAfter": self._n_sentences_after,
                    },
                    "corpusKey": corpus_keys,
                }
            ]
        }
        if self._rerank:
            reranking_config = {
                "rerankerId": self._reranker_id,
            }
            if self._reranker_id == MMR_RERANKER_ID:
                reranking_config["mmrConfig"] = {
                    "diversityBias": self._mmr_diversity_bias
                }
            elif self._reranker_id == UDF_RERANKER_ID:
                reranking_config["userFunction"] = self._udf_expression

            data["query"][0]["rerankingConfig"] = reranking_config

        if self._summary_enabled:
            summary_config = {
                "responseLang": self._summary_response_lang,
                "maxSummarizedResults": self._summary_num_results,
                "summarizerPromptName": self._summary_prompt_name,
            }
            data["query"][0]["summary"] = [summary_config]
            if chat:
                data["query"][0]["summary"][0]["chat"] = {
                    "store": True,
                    "conversationId": chat_conv_id,
                }

            if self._citations_style:
                if self._citations_style in ["NUMERIC", "NONE"]:
                    data["query"][0]["summary"][0]["citationParams"] = {
                        "style": self._citations_style,
                    }

                elif self._citations_url_pattern:
                    data["query"][0]["summary"][0]["citationParams"] = {
                        "style": self._citations_style,
                        "urlPattern": self._citations_url_pattern,
                        "textPattern": self._citations_text_pattern,
                    }

        return data

    def _vectara_stream(
        self,
        query_bundle: QueryBundle,
        chat: bool = False,
        conv_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TokenGen:
        """
        Query Vectara index to get for top k most similar nodes.

        Args:
            query_bundle: Query Bundle
            chat: whether to enable chat
            conv_id: conversation ID, if chat enabled
        """
        body = self._build_vectara_query_body(query_bundle.query_str)
        response = self._index._session.post(
            headers=self._get_post_headers(),
            url="https://api.vectara.io/v1/stream-query",
            data=json.dumps(body),
            timeout=self._index.vectara_api_timeout,
            stream=True,
        )

        if response.status_code != 200:
            print(
                "Query failed %s",
                f"(code {response.status_code}, reason {response.reason}, details "
                f"{response.text})",
            )
            return

        responses = []
        documents = []
        stream_response = CompletionResponse(
            text="", additional_kwargs={"fcs": None}, raw=None, delta=None
        )

        for line in response.iter_lines():
            if line:  # filter out keep-alive new lines
                data = json.loads(line.decode("utf-8"))
                result = data["result"]
                response_set = result["responseSet"]
                if response_set is None:
                    summary = result.get("summary", None)
                    if summary is None:
                        continue
                    if len(summary.get("status")) > 0:
                        print(
                            f"Summary generation failed with status {summary.get('status')[0].get('statusDetail')}"
                        )
                        continue

                    # Store conversation ID for chat, if applicable
                    chat = summary.get("chat", None)
                    if chat and chat.get("status", None):
                        st_code = chat["status"]
                        print(f"Chat query failed with code {st_code}")
                        if st_code == "RESOURCE_EXHAUSTED":
                            self.conv_id = None
                            print("Sorry, Vectara chat turns exceeds plan limit.")
                            continue

                    conv_id = chat.get("conversationId", None) if chat else None
                    if conv_id:
                        self.conv_id = conv_id

                    # if factual consistency score is provided, pull that from the JSON response
                    if summary.get("factualConsistency", None):
                        fcs = summary.get("factualConsistency", {}).get("score", None)
                        stream_response.additional_kwargs["fcs"] = fcs
                        continue

                    # Yield the summary chunk
                    chunk = urllib.parse.unquote(summary["text"])
                    stream_response.text += chunk
                    stream_response.delta = chunk
                    yield stream_response
                else:
                    metadatas = []
                    for x in responses:
                        md = {m["name"]: m["value"] for m in x["metadata"]}
                        doc_num = x["documentIndex"]
                        doc_md = {
                            m["name"]: m["value"]
                            for m in documents[doc_num]["metadata"]
                        }
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
                    stream_response.additional_kwargs["top_nodes"] = top_nodes[
                        : self._similarity_top_k
                    ]
                    stream_response.delta = None
                    yield stream_response
        return

    def _vectara_query(
        self,
        query_bundle: QueryBundle,
        chat: bool = False,
        conv_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[NodeWithScore], Dict, str]:
        """
        Query Vectara index to get for top k most similar nodes.

        Args:
            query: Query Bundle
            Additional keyword arguments

        Returns:
            List[NodeWithScore]: list of nodes with scores
            Dict: summary
            str: conversation ID, if applicable
        """
        data = self._build_vectara_query_body(query_bundle.query_str, chat, conv_id)

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
            return [], {"text": ""}, ""

        result = response.json()
        status = result["responseSet"][0]["status"]
        if len(status) > 0 and status[0]["code"] != "OK":
            _logger.error(
                f"Query failed (code {status[0]['code']}, msg={status[0]['statusDetail']}"
            )
            return [], {"text": ""}, ""

        responses = result["responseSet"][0]["response"]
        documents = result["responseSet"][0]["document"]

        if self._summary_enabled:
            summaryJson = result["responseSet"][0]["summary"][0]
            if len(summaryJson["status"]) > 0:
                print(
                    f"Summary generation failed with error: '{summaryJson['status'][0]['statusDetail']}'"
                )
                return [], {"text": ""}, ""

            summary = {
                "text": (
                    urllib.parse.unquote(summaryJson["text"])
                    if self._summary_enabled
                    else None
                ),
                "fcs": summaryJson["factualConsistency"]["score"],
            }
            if summaryJson.get("chat", None):
                conv_id = summaryJson["chat"]["conversationId"]
            else:
                conv_id = None
        else:
            summary = None

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

        return top_nodes[: self._similarity_top_k], summary, conv_id

    async def _avectara_query(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], Dict]:
        """
        Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return await self._vectara_query(query_bundle)


class VectaraAutoRetriever(VectorIndexAutoRetriever):
    """
    Managed Index auto retriever.

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
