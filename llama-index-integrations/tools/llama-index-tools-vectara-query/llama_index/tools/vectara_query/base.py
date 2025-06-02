from typing import Any, Union, List, Dict, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.schema import QueryBundle
from llama_index.core.callbacks.base import CallbackManager

from llama_index.indices.managed.vectara import VectaraIndex
from llama_index.indices.managed.vectara.retriever import VectaraRetriever
from llama_index.indices.managed.vectara.query import VectaraQueryEngine


class VectaraQueryToolSpec(BaseToolSpec):
    """Vectara Query tool spec."""

    spec_functions = ["semantic_search", "rag_query"]

    def __init__(
        self,
        vectara_corpus_key: Optional[str] = None,
        vectara_api_key: Optional[str] = None,
        num_results: int = 5,
        offset: int = 0,
        lambda_val: Union[List[float], float] = 0.005,
        semantics: Union[List[str], str] = "default",
        custom_dimensions: Union[List[Dict], Dict] = {},
        n_sentences_before: int = 2,
        n_sentences_after: int = 2,
        metadata_filter: Union[List[str], str] = "",
        reranker: str = "mmr",
        rerank_k: int = 50,
        rerank_limit: Optional[int] = None,
        rerank_cutoff: Optional[float] = None,
        mmr_diversity_bias: float = 0.2,
        udf_expression: str = None,
        rerank_chain: List[Dict] = None,
        summarizer_prompt_name: str = "vectara-summary-ext-24-05-sml",
        summary_num_results: int = 5,
        summary_response_lang: str = "eng",
        prompt_text: Optional[str] = None,
        max_response_chars: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        citations_style: Optional[str] = None,
        citations_url_pattern: Optional[str] = None,
        citations_text_pattern: Optional[str] = None,
        save_history: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Vectara API and query parameters.

        Parameters
        ----------
        - vectara_corpus_key (str): The corpus key for the corpus you want to search for information.
            If not specified, reads for environment variable "VECTARA_CORPUS_KEY".
        - vectara_api_key (str): An API key that has query permissions for the given corpus.
            If not specified, reads for environment variable "VECTARA_API_KEY".
        - num_results (int): Number of search results to return with response.
        - offset (int): Number of results to skip.
        - lambda_val (Union[List[float], float]): Lambda value for the Vectara query.
            Provide single value for one corpus or a list of values for each corpus.
        - semantics (Union[List[str], str]): Indicates whether the query is intended as a query or response.
            Provide single value for one corpus or a list of values for each corpus.
        - custom_dimensions (Dict): Custom dimensions for the query.
            See (https://docs.vectara.com/docs/learn/semantic-search/add-custom-dimensions)
            for more details about usage.
            Provide single dict for one corpus or a list of dicts for each corpus.
        - n_sentences_before (int): Number of sentences before the summary.
        - n_sentences_after (int): Number of sentences after the summary.
        - metadata_filter (Union[List[str], str]): A string with expressions to filter the search documents for each corpus.
            Provide single string for one corpus or a list of strings for each corpus (if multiple corpora).
        - reranker (str): The reranker to use, either mmr, slingshot (i.e. multilingual_reranker_v1), userfn, or chain.
        - rerank_k (int): Number of top-k documents for reranking.
        - rerank_limit (int): maximum number of results to return after reranking, defaults to 50.
            Don't specify this for chain reranking. Instead, put the "limit" parameter in the dict for each individual reranker.
        - rerank_cutoff (float): minimum score threshold for results to include after reranking, defaults to 0.
            Don't specify this for chain reranking. Instead, put the "chain" parameter in the dict for each individual reranker.
        - mmr_diversity_bias (float): MMR diversity bias.
        - udf_expression (str): the user defined expression for reranking results.
            See (https://docs.vectara.com/docs/learn/user-defined-function-reranker)
            for more details about syntax for udf reranker expressions.
        - rerank_chain: a list of rerankers to be applied in a sequence and their associated parameters
            for the chain reranker. Each element should specify the "type" of reranker (mmr, slingshot, userfn)
            and any other parameters (e.g. "limit" or "cutoff" for any type,  "diversity_bias" for mmr, and "user_function" for udf).
            If using slingshot/multilingual_reranker_v1, it must be first in the list.
        - summarizer_prompt_name (str): If enable_summarizer is True, the Vectara summarizer to use.
        - summary_num_results (int): If enable_summarizer is True, the number of summary results.
        - summary_response_lang (str): If enable_summarizer is True, the response language for the summary.
        - prompt_text (str): the custom prompt, using appropriate prompt variables and functions.
            See (https://docs.vectara.com/docs/1.0/prompts/custom-prompts-with-metadata)
            for more details.
        - max_response_chars (int): the desired maximum number of characters for the generated summary.
        - max_tokens (int): the maximum number of tokens to be returned by the LLM.
        - temperature (float): The sampling temperature; higher values lead to more randomness.
        - frequency_penalty (float): How much to penalize repeating tokens in the response, reducing likelihood of repeating the same line.
        - presence_penalty (float): How much to penalize repeating tokens in the response, increasing the diversity of topics.
        - citations_style (str): The style of the citations in the summary generation,
            either "numeric", "html", "markdown", or "none". Defaults to None.
        - citations_url_pattern (str): URL pattern for html and markdown citations.
            If non-empty, specifies the URL pattern to use for citations; e.g. "{doc.url}".
            See (https://docs.vectara.com/docs/api-reference/search-apis/search#citation-format-in-summary) for more details.
            Defaults to None.
        - citations_text_pattern (str): The displayed text for citations.
            If not specified, numeric citations are displayed.
        - save_history (bool): Whether to save the query in history. Defaults to False.

        """
        self.index = VectaraIndex(
            vectara_corpus_key=vectara_corpus_key,
            vectara_api_key=vectara_api_key,
        )

        self.retriever = VectaraRetriever(
            index=self.index,
            similarity_top_k=num_results,
            offset=offset,
            lambda_val=lambda_val,
            semantics=semantics,
            custom_dimensions=custom_dimensions,
            n_sentences_before=n_sentences_before,
            n_sentences_after=n_sentences_after,
            filter=metadata_filter,
            reranker=reranker,
            rerank_k=rerank_k,
            rerank_limit=rerank_limit,
            rerank_cutoff=rerank_cutoff,
            mmr_diversity_bias=mmr_diversity_bias,
            udf_expression=udf_expression,
            rerank_chain=rerank_chain,
            summary_enabled=False,
            callback_manager=callback_manager,
            **kwargs,
        )

        query_engine_retriever = VectaraRetriever(
            index=self.index,
            similarity_top_k=num_results,
            offset=offset,
            lambda_val=lambda_val,
            semantics=semantics,
            custom_dimensions=custom_dimensions,
            n_sentences_before=n_sentences_before,
            n_sentences_after=n_sentences_after,
            filter=metadata_filter,
            reranker=reranker,
            rerank_k=rerank_k,
            rerank_limit=rerank_limit,
            rerank_cutoff=rerank_cutoff,
            mmr_diversity_bias=mmr_diversity_bias,
            udf_expression=udf_expression,
            rerank_chain=rerank_chain,
            summary_enabled=True,
            summary_response_lang=summary_response_lang,
            summary_num_results=summary_num_results,
            summary_prompt_name=summarizer_prompt_name,
            prompt_text=prompt_text,
            max_response_chars=max_response_chars,
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
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

        Parameters
        ----------
            query (str): The input query from the user.

        Returns
        -------
            List[Dict]: A list of retrieved documents with their associated metadata

        """
        response = self.retriever._retrieve(query_bundle=QueryBundle(query_str=query))

        if len(response) == 0:
            return []

        return [
            {
                "text": doc.node.text_resource.text,
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

        Parameters
        ----------
            query (str): The input query from the user.

        Returns
        -------
            Dict: A dictionary containing the generated summary, citation metadata, and the factual consistency score.

        """
        response = self.query_engine._query(query_bundle=QueryBundle(query_str=query))

        if str(response) == "None":
            return {}

        return {
            "summary": response.response,
            "citation_metadata": response.source_nodes,
            "factual_consistency_score": response.metadata["fcs"]
            if "fcs" in response.metadata
            else 0.0,
        }
