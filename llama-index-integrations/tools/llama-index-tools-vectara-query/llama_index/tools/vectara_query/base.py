from typing import Any, List, Dict, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.schema import QueryBundle
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks.base import CallbackManager

from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.indices.managed.vectara.retriever import VectaraRetriever
from llama_index.indices.managed.vectara.query import VectaraQueryEngine


class VectaraQueryToolSpec(BaseToolSpec):
    """Vectara Query tool spec."""

    spec_functions = ["semantic_search", "rag_query"]

    def __init__(
        self,
        vectara_customer_id: str,
        vectara_corpus_id: str,
        vectara_api_key: str,
        num_results: int = 5,
        lambda_val: float = 0.005,
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        metadata_filter: str = "",
        reranker: str = "mmr",
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.2,
        udf_expression: str = None,
        summarizer_prompt_name: str = "vectara-summary-ext-24-05-sml",
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        citations_style: Optional[str] = None,
        citations_url_pattern: Optional[str] = None,
        citations_text_pattern: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Vectara API and query parameters.

        Parameters:
        - vectara_customer_id (str): Your Vectara customer ID.
        - vectara_corpus_id (str): The corpus ID for the corpus you want to search for information.
        - vectara_api_key (str): An API key that has query permissions for the given corpus.
        - num_results (int): Number of search results to return with response.
        - lambda_val (float): Lambda value for the Vectara query.
        - n_sentences_before (int): Number of sentences before the summary.
        - n_sentences_after (int): Number of sentences after the summary.
        - metadata_filter (str): A string with expressions to filter the search documents.
        - reranker (str): The reranker mode, either "mmr", "slingshot", "multilingual_reranker_v1", "udf", or "none".
        - rerank_k (int): Number of top-k documents for reranking.
        - mmr_diversity_bias (float): MMR diversity bias.
        - udf_expression (str): the user defined expression for reranking results.
            See (https://docs.vectara.com/docs/learn/user-defined-function-reranker)
            for more details about syntax for udf reranker expressions.
        - summarizer_prompt_name (str): If enable_summarizer is True, the Vectara summarizer to use.
        - summary_num_results (int): If enable_summarizer is True, the number of summary results.
        - summary_response_lang (str): If enable_summarizer is True, the response language for the summary.
        - citations_style (str): The style of the citations in the summary generation,
            either "numeric", "html", "markdown", or "none".
            This is a Vectara Scale only feature. Defaults to None.
        - citations_url_pattern (str): URL pattern for html and markdown citations.
            If non-empty, specifies the URL pattern to use for citations; e.g. "{doc.url}".
            See (https://docs.vectara.com/docs/api-reference/search-apis/search#citation-format-in-summary) for more details.
            This is a Vectara Scale only feature. Defaults to None.
        - citations_text_pattern (str): The displayed text for citations.
            Must be specified for html and markdown citations.
        """
        self.index = VectaraIndex(
            vectara_customer_id=vectara_customer_id,
            vectara_corpus_id=vectara_corpus_id,
            vectara_api_key=vectara_api_key,
        )

        self.retriever = VectaraRetriever(
            index=self.index,
            similarity_top_k=num_results,
            lambda_val=lambda_val,
            n_sentences_before=n_sentences_before,
            n_sentences_after=n_sentences_after,
            filter=metadata_filter,
            reranker=reranker,
            rerank_k=rerank_k,
            mmr_diversity_bias=mmr_diversity_bias,
            udf_expression=udf_expression,
            summary_enabled=False,
            callback_manager=callback_manager,
            **kwargs,
        )

        query_engine_retriever = VectaraRetriever(
            index=self.index,
            similarity_top_k=num_results,
            lambda_val=lambda_val,
            n_sentences_before=n_sentences_before,
            n_sentences_after=n_sentences_after,
            filter=metadata_filter,
            reranker=reranker,
            rerank_k=rerank_k,
            mmr_diversity_bias=mmr_diversity_bias,
            udf_expression=udf_expression,
            summary_enabled=True,
            summary_response_lang=summary_response_lang,
            summary_num_results=summary_num_results,
            summary_prompt_name=summarizer_prompt_name,
            citations_style=citations_style,
            citations_url_pattern=citations_url_pattern,
            citations_text_pattern=citations_text_pattern,
            callback_manager=callback_manager,
            **kwargs,
        )

        self.query_engine = VectaraQueryEngine(retriever=query_engine_retriever)

    def semantic_search(
        self,
        query: str,
    ) -> List[Dict]:
        """
        Makes a query to a Vectara corpus and returns the top search results from the retrieved documents.

        Parameters:
        - query (str): The input query from the user.
        """
        response = self.retriever._retrieve(query_bundle=QueryBundle(query_str=query))

        if len(response) == 0:
            return Response(response="Tool failed to retrieve any documents.")

        return [
            {
                "text": doc.node.text,
                "citation_metadata": doc.node.metadata,
            }
            for doc in response
        ]

    def rag_query(
        self,
        query: str,
    ) -> Dict:
        """
        Makes a query to a Vectara corpus and returns the generated summary, the citation metadata, and the factual consistency score.

        Parameters:
        - query (str): The input query from the user.
        """
        response = self.query_engine._query(query_bundle=QueryBundle(query_str=query))

        if str(response) == "None":
            return Response(response="Tool failed to generate a response.")

        return {
            "summary": response.response,
            "citation_metadata": response.source_nodes,
            "factual_consistency_score": response.metadata["fcs"]
            if "fcs" in response.metadata
            else 0.0,
        }
