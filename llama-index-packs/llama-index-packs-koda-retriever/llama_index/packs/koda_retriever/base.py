"""
Defines a custom hybrid retriever for flexible alpha tuning.

Idea & original implementation sourced from the following docs:
    - https://blog.llamaindex.ai/llamaindex-enhancing-retrieval-performance-with-alpha-tuning-in-hybrid-search-in-rag-135d0c9b8a00
    - https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167

Author: no_dice
"""

from typing import Optional, List, Any
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore, QueryType, QueryBundle
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.base.response.schema import RESPONSE_TYPE
from .constants import CATEGORIZER_PROMPT, DEFAULT_CATEGORIES
import logging
from .matrix import AlphaMatrix


class KodaRetriever(BaseRetriever):
    """
    Custom Hybrid Retriever that dynamically determines the optimal alpha for a given query.
    An LLM is used to categorize the query and therefore determine the optimal alpha value, as each category has a preset/provided alpha value.
    It is recommended that you run tests on your corpus of data and queries to determine categories and corresponding alpha values for your use case.

    KodaRetriever is built from BaseRetriever, and therefore is a llama-index compatible drop-in replacement for any hybrid retriever.

    Auto-routing is NOT enabled without providing an LLM.
    If no LLM is provided, the default alpha value will be used for all queries and no alpha tuning will be done.
    Reranking will be done automatically if a reranker is provided.
    If no matrix is provided, a default matrix is leveraged. (Not recommended for production use)

    Args:
        index (VectorStoreIndex): The index to be used for retrieval
        llm (LLM, optional): The LLM to be used for auto-routing. Defaults to None.
        reranker (BaseNodePostprocessor, optional): The reranker to be used for postprocessing. Defaults to None.
        default_alpha (float, optional): The default alpha value to be used if no LLM is provided. Defaults to .5.
        matrix (dict or AlphaMatrix, optional): The matrix to be used for auto-routing. Defaults to AlphaMatrix(data=DEFAULT_CATEGORIES).
        verbose (bool, optional): Whether to log verbose output. Defaults to False.
        **kwargs: Additional arguments for VectorIndexRetriever

    Returns:
        KodaRetriever

    Examples:
        >>> # Example 1 - provide your own LLM
        >>> retriever = KodaRetriever( # woof woof
                            index=vector_index
                            , llm=Settings.llm
                            , verbose=True
                        )
        >>> results = retriever.retrieve("What is the capital of France?")

        >>> # Example 2 - set custom alpha values
        >>> matrix_data = { # this is just dummy data, alpha values were randomly chosen
                "positive sentiment": {
                    "alpha": .2
                    , "description": "Queries expecting a positive answer"
                    , "examples": [
                        "I love this product"
                        , "This is the best product ever"
                    ]
                }
                , "negative sentiment": {
                    "alpha": .7
                    , "description": "Queries expecting a negative answer"
                    , "examples": [
                        "I hate this product"
                        , "This is the worst product ever"
                    ]
                }
            }

        >>> retriever = KodaRetriever( # woof woof
                            index=vector_index
                            , llm=Settings.llm
                            , matrix=matrix_data
                            , verbose=True
                        )
        >>> results = retriever.retrieve("What happened on Y2K?")

    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: Optional[LLM] = None,  # if I could, I'd default to
        reranker: Optional[BaseNodePostprocessor] = None,
        default_alpha: float = 0.5,
        matrix: dict or AlphaMatrix = DEFAULT_CATEGORIES,  # type: ignore
        verbose: bool = False,
        **kwargs,  # kwargs for VectorIndexRetriever
    ):
        super().__init__()

        self.index = index
        self.retriever = VectorIndexRetriever(
            index=index,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=default_alpha,
            **kwargs,  # filters, etc, added here
        )
        self.default_alpha = default_alpha
        self.reranker = reranker
        self.llm = llm
        self.verbose = verbose

        if isinstance(matrix, dict):
            matrix = AlphaMatrix(data=matrix)

        self.matrix = matrix

    def categorize(self, query: str) -> str:
        """Categorizes a query using the provided LLM and matrix. If no LLM is provided, the default alpha value will be used."""
        if not self.llm:
            err = "LLM is required for auto-routing. During instantiation, please provide an LLM or use direct routing."
            raise TypeError(err)

        prompt = CATEGORIZER_PROMPT.format(
            question=query, category_info=self.matrix.get_all_category_info()
        )

        response = str(self.llm.complete(prompt))  # type: ignore

        if response not in self.matrix.get_categories():
            raise ValueError(
                f"LLM classified question in a category that is not registered. {response} not in {self.matrix.get_categories()}"
            )

        return response

    async def a_categorize(self, query: str) -> str:
        """(async) Categorizes a query using the provided LLM and matrix. If no LLM is provided, the default alpha value will be used."""
        if not self.llm:
            err = "LLM is required for auto-routing. During instantiation, please provide an LLM or use direct routing."
            raise TypeError(err)

        prompt = CATEGORIZER_PROMPT.format(
            question=query, category_info=self.matrix.get_all_category_info()
        )

        response = await self.llm.acomplete(prompt)  # type: ignore
        response = str(response)

        if response not in self.matrix.get_categories():
            raise ValueError(
                f"LLM classified question in a category that is not registered. {response!s} not in {self.matrix.get_categories()}"
            )

        return response

    def category_retrieve(self, category: str, query: QueryType) -> List[NodeWithScore]:
        """Updates the alpha and retrieves results for a query using the provided category and query. If no LLM is provided, the default alpha value will be used."""
        alpha = self.matrix.get_alpha(category)
        self.retriever._alpha = (
            alpha  # updates alpha according to classification of query
        )

        return self.retriever.retrieve(str_or_query_bundle=query)

    async def a_category_retrieve(
        self, category: str, query: QueryType
    ) -> List[NodeWithScore]:
        """(async) Updates the alpha and retrieves results for a query using the provided category and query. If no LLM is provided, the default alpha value will be used."""
        alpha = self.matrix.get_alpha(category)
        self.retriever._alpha = (
            alpha  # updates alpha according to classification of query
        )

        return await self.retriever.aretrieve(str_or_query_bundle=query)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """llama-index compatible retrieve method that auto-determines the optimal alpha for a query and then retrieves results for a query."""
        if not self.llm:
            warning = f"LLM is not provided, skipping route categorization. Default alpha of {self.default_alpha} will be used."
            logging.warning(warning)

            results = self.retriever.retrieve(query_bundle)
        else:
            category = self.categorize(query=query_bundle.query_str)  # type: ignore
            results = self.category_retrieve(category, query_bundle)
            if self.verbose:
                logging.info(
                    f"Query categorized as {category} with alpha of {self.matrix.get_alpha(category)}"
                )

        if self.reranker:
            if self.verbose:
                logging.info("Reranking results")
            results = self.reranker.postprocess_nodes(
                nodes=results, query_bundle=query_bundle
            )

        return results

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """(async) llama-index compatible retrieve method that auto-determines the optimal alpha for a query and then retrieves results for a query."""
        if not self.llm:
            warning = f"LLM is not provided, skipping route categorization. Default alpha of {self.default_alpha} will be used."
            logging.warning(warning)

            results = await self.retriever.aretrieve(query_bundle)
        else:
            category = await self.a_categorize(query_bundle.query_str)  # type: ignore
            results = await self.a_category_retrieve(category, query_bundle)
            if self.verbose:
                logging.info(
                    f"Query categorized as {category} with alpha of {self.matrix.get_alpha(category)}"
                )

        if self.reranker:
            if self.verbose:
                logging.info("Reranking results")
            results = self.reranker.postprocess_nodes(
                nodes=results, query_bundle=query_bundle
            )

        return results


class KodaRetrieverPack(BaseLlamaPack):
    def __init__(
        self,
        index: VectorStoreIndex,
        llm: Optional[LLM] = None,  # if I could, I'd default to
        reranker: Optional[BaseNodePostprocessor] = None,
        default_alpha: float = 0.5,
        matrix: dict or AlphaMatrix = DEFAULT_CATEGORIES,  # type: ignore
        verbose: bool = False,
        **kwargs,  # kwargs for VectorIndexRetriever
    ) -> None:
        """Init params."""
        self.retriever = KodaRetriever(
            index=index,
            llm=llm,
            reranker=reranker,
            default_alpha=default_alpha,
            matrix=matrix,
            verbose=verbose,
            **kwargs,
        )

    def get_modules(self) -> dict:
        """Get modules."""
        return {
            "retriever": self.retriever,
            "retriever_cls": KodaRetriever,
        }

    def run(self, query_str: str, **kwargs: Any) -> RESPONSE_TYPE:
        """Run method."""
        return self.retriever.retrieve(query_str, **kwargs)

    async def arun(self, query_str: str, **kwargs: Any) -> RESPONSE_TYPE:
        """Asynchronous run method."""
        return await self.retriever.aretrieve(query_str, **kwargs)
