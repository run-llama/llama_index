from typing import List, Dict
import re
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.base.response.schema import Response
from llama_index.indices.managed.vectara import VectaraIndex


class VectaraQueryToolSpec(BaseToolSpec):
    """Vectara Query tool spec."""

    spec_functions = ["rag_query", "semantic_search"]

    def __init__(
        self, vectara_customer_id: str, vectara_corpus_id: str, vectara_api_key: str
    ) -> None:
        """Initialize the Vectara API."""
        self.vectara_customer_id = vectara_customer_id
        self.vectara_corpus_id = vectara_corpus_id
        self.vectara_api_key = vectara_api_key

        self.index = VectaraIndex(
            vectara_customer_id=self.vectara_customer_id,
            vectara_corpus_id=self.vectara_corpus_id,
            vectara_api_key=self.vectara_api_key,
        )

    def rag_query(
        self,
        query: str,
        summarizer_prompt_name: str = "vectara-summary-ext-24-05-sml",
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        metadata_filter: str = "",
        lambda_val: float = 0.005,
        reranker: str = "mmr",
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.2,
        include_citations: bool = True,
    ) -> Dict:
        """
        Makes a query to a Vectara corpus and returns the generated summary and associated metadata.

        Parameters:
        - query (str): The input query from the user.
        - summarizer_prompt_name (str): The Vectara summarizer to use.
        - summary_num_results (int): The number of summary results.
        - summary_response_lang (str): The response language for the summary.
        - n_sentences_before (int): Number of sentences before the summary.
        - n_sentences_after (int): Number of sentences after the summary.
        - metadata_filter (str): A string with expressions to filter the search documents.
        - lambda_val (float): Lambda value for the Vectara query.
        - reranker (str): The reranker mode, either "mmr" or "default". # ASK OFER WHAT DEFAULT WILL DO.
        - rerank_k (int): Number of top-k documents for reranking.
        - mmr_diversity_bias (float): MMR diversity bias.
        - include_citations (bool): Whether to include citations in the response.
          If True, uses MARKDOWN vectara citations that requires the Vectara scale plan.
        """
        query_engine = self.index.as_query_engine(
            lambda_val=lambda_val,
            n_sentences_before=n_sentences_before,
            n_sentences_after=n_sentences_after,
            filter=metadata_filter,
            vectara_query_mode=reranker,
            rerank_k=rerank_k,
            mmr_diversity_bias=mmr_diversity_bias,
            summary_enabled=True,
            summary_response_lang=summary_response_lang,
            summary_num_results=summary_num_results,
            summary_prompt_name=summarizer_prompt_name,
        )

        response = query_engine.query(query)

        if str(response) == "None":
            return Response("Tool failed to generate a response.")

        # Extract citation metadata
        pattern = r"\[\[(\d+)\]" if include_citations else r"\[(\d+)\]"
        matches = re.findall(pattern, response.response)
        citation_numbers = [int(match) for match in matches]
        citation_metadata: dict = {
            f"metadata for citation {citation_number}": response.source_nodes[
                citation_number - 1
            ].metadata
            for citation_number in citation_numbers
        }
        return {
            "response": response.response,
            "citation_metadata": citation_metadata,
            "factual_consistency": response.metadata["fcs"]
            if "fcs" in response.metadata
            else 0.0,
        }

    def semantic_search(
        self,
        query: str,
        num_results: int = 5,
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        metadata_filter: str = "",
        lambda_val: float = 0.005,
        reranker: str = "mmr",
        rerank_k: int = 50,
        mmr_diversity_bias: float = 0.2,
    ) -> List[Dict]:
        """
        Makes a query to a Vectara corpus and returns the top search results from the retrieved documents.

        Parameters:
        - query (str): The input query from the user.
        - num_results (int): Number of search results to return
        - n_sentences_before (int): Number of sentences before the summary.
        - n_sentences_after (int): Number of sentences after the summary.
        - metadata_filter (str): A string with expressions to filter the search documents.
        - lambda_val (float): Lambda value for the Vectara query.
        - reranker (str): The reranker mode, either "mmr" or "default".
        - rerank_k (int): Number of top-k documents for reranking.
        - mmr_diversity_bias (float): MMR diversity bias.
        """
        retriever_engine = self.index.as_retriever(
            lambda_val=lambda_val,
            similarity_top_k=num_results,
            n_sentences_before=n_sentences_before,
            n_sentences_after=n_sentences_after,
            filter=metadata_filter,
            vectara_query_mode=reranker,
            rerank_k=rerank_k,
            mmr_diversity_bias=mmr_diversity_bias,
        )

        response = retriever_engine.retrieve(query)

        res = []
        for doc in response:
            res.append(
                {"text": doc.node.text, "metadata": doc.node.metadata, "FCS": doc.score}
            )
        return res
