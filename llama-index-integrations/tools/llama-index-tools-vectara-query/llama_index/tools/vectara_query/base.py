from typing import List, Dict, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.schema import QueryBundle
from llama_index.core.base.response.schema import Response

from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.indices.managed.vectara.retriever import VectaraRetriever
from llama_index.indices.managed.vectara.query import VectaraQueryEngine


## FOLLOW THIS FOR BUILDING TOOLS: https://github.com/run-llama/llama_index/blob/15227173b8c1241c9fbc761342a2344cd90c6593/llama-index-integrations/indices/llama-index-indices-managed-vectara/llama_index/indices/managed/vectara/query.py#L159


class VectaraQueryToolSpec(BaseToolSpec):
    """Vectara Query tool spec."""

    spec_functions = ["rag_query", "semantic_search"]

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
        summarizer_prompt_name: str = "vectara-summary-ext-24-05-sml",
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        citations_pattern: Optional[
            str
        ] = None,  # Currently can only use MARKDOWN for citation type and does not allow user to specify text pattern (this is probably okay, but just want to make sure that Ofer is okay with this).
        # See _build_vectara_query_body function at https://github.com/run-llama/llama_index/blob/3327c92fcb6c7dddc6d4b8ed7da30d88acbaf852/llama-index-integrations/indices/llama-index-indices-managed-vectara/llama_index/indices/managed/vectara/retriever.py#L223
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
        - reranker (str): The reranker mode, either "mmr", "multilingual_reranker_v1", or "none".
        - rerank_k (int): Number of top-k documents for reranking.
        - mmr_diversity_bias (float): MMR diversity bias.
        - summarizer_prompt_name (str): If enable_summarizer is True, the Vectara summarizer to use.
        - summary_num_results (int): If enable_summarizer is True, the number of summary results.
        - summary_response_lang (str): If enable_summarizer is True, the response language for the summary.
        - citations_pattern (str): URL pattern for citations. If non-empty, specifies
            the URL pattern to use for citations; for example "{doc.url}".
            see (https://docs.vectara.com/docs/api-reference/search-apis/search#citation-format-in-summary) for more details.
            If unspecified, citations are generated in numeric form [1],[2], etc
            This is a Vectara Scale only feature. Defaults to None.
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
            summary_enabled=True,
            summary_response_lang=summary_response_lang,
            summary_num_results=summary_num_results,
            summary_prompt_name=summarizer_prompt_name,
            citations_url_pattern=citations_pattern,
        )

        self.query_engine = VectaraQueryEngine(retriever=self.retriever)

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

        # print(f"DEBUG: GOT RESPONSE FROM QUERY: {response}")

        return {
            "summary": response.response,
            "citation_metadata": response.source_nodes,
            "factual_consistency_score": response.metadata["fcs"]
            if "fcs" in response.metadata
            else 0.0,
        }

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

        # print(f"DEBUG: GOT RESPONSE FROM QUERY: {response}")

        if len(response) == 0:
            return Response(response="Tool failed to retrieve any documents.")

        return [
            {
                "text": doc.node.text,
                "citation_metadata": doc.node.metadata,
                "factual_consistency_score": doc.score,
            }
            for doc in response
        ]
