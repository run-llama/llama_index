"""Query transform."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, cast

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.indices.query.query_transform.prompts import (
    DEFAULT_COT_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    CoTDecomposeQueryTransformPrompt,
    DecomposeQueryTransformPrompt,
)
from gpt_index.indices.query.schema import QueryBundle, QueryConfig
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT
from gpt_index.response.schema import Response


class BaseQueryTransform:
    """Base class for query transform.

    A query transform augments a raw query string with associated transformations
    to improve index querying.

    The query transformation is performed before the query is sent to the index.

    """

    @abstractmethod
    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""

    def run(
        self,
        query_bundle_or_str: Union[str, QueryBundle],
        extra_info: Optional[str] = None,
    ) -> QueryBundle:
        """Run query transform."""
        extra_info = extra_info or {}
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(
                query_str=query_bundle, custom_embedding_strs=[query_bundle]
            )
        else:
            query_bundle = query_bundle_or_str

        return self._run(query_bundle, extra_info=extra_info)

    def __call__(
        self,
        query_bundle_or_str: Union[str, QueryBundle],
        extra_info: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query processor."""
        return self.run(query_bundle_or_str)


class IdentityQueryTransform(BaseQueryTransform):
    """Identity query transform.

    Do nothing to the query.

    """

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        return query_bundle


class HyDEQueryTransform(BaseQueryTransform):
    """Hypothetical Document Embeddings (HyDE) query transform.

    It uses an LLM to generate hypothetical answer(s) to a given query,
    and use the resulting documents as embedding strings.

    As described in `[Precise Zero-Shot Dense Retrieval without Relevance Labels]
    (https://arxiv.org/abs/2212.10496)`
    """

    def __init__(
        self,
        llm_predictor: Optional[LLMPredictor] = None,
        hyde_prompt: Optional[Prompt] = None,
        include_original: bool = True,
    ) -> None:
        """Initialize HyDEQueryTransform.

        Args:
            llm_predictor (Optional[LLMPredictor]): LLM for generating
                hypothetical documents
            hyde_prompt (Optional[Prompt]): Custom prompt for HyDE
            include_original (bool): Whether to include original query
                string as one of the embedding strings
        """
        super().__init__()

        self._llm_predictor = llm_predictor or LLMPredictor()
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
        self._include_original = include_original

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        # TODO: support generating multiple hypothetical docs
        query_str = query_bundle.query_str
        hypothetical_doc, _ = self._llm_predictor.predict(
            self._hyde_prompt, context_str=query_str
        )
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.extend(query_bundle.embedding_strs)
        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs,
        )


class DecomposeQueryTransform(BaseQueryTransform):
    """Decompose query transform.

    Decomposes query into a subquery given the current index struct.
    Performs a single step transformation.

    Args:
        llm_predictor (Optional[LLMPredictor]): LLM for generating
            hypothetical documents

    """

    def __init__(
        self,
        llm_predictor: Optional[LLMPredictor] = None,
        decompose_query_prompt: Optional[DecomposeQueryTransformPrompt] = None,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._decompose_query_prompt = (
            decompose_query_prompt or DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        index_struct = cast(IndexStruct, extra_info.get("index_struct", None))
        # currently, just get text from the index
        index_text = index_struct.get_text()

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        new_query_str, _ = self._llm_predictor.predict(
            self._decompose_query_prompt,
            query_str=query_str,
            context_str=index_text,
        )
        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
        )


class CoTDecomposeQueryTransform(BaseQueryTransform):
    """Chain of thought decompose query transform.

    Decomposes query into multiple subqueries given index struct.

    NOTE: doesn't work yet.

    Args:
        llm_predictor (Optional[LLMPredictor]): LLM for generating
            hypothetical documents

    """

    def __init__(
        self,
        llm_predictor: Optional[LLMPredictor] = None,
        cot_decompose_query_prompt: Optional[CoTDecomposeQueryTransformPrompt] = None,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._cot_decompose_query_prompt = (
            cot_decompose_query_prompt or DEFAULT_COT_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> List[QueryBundle]:
        """Run query transform."""
        index_struct = cast(IndexStruct, extra_info.get("index_struct", None))
        prev_response = cast(Response, extra_info.get("prev_response"), None)
        # currently, just get text from the index
        index_text = index_struct.get_text()

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        new_query_str, _ = self._llm_predictor.predict(
            self._cot_decompose_query_prompt,
            prev_reasoning=prev_response,
            query_str=query_str,
            context_str=index_text,
        )
        return [
            QueryBundle(
                query_str=new_query_str,
                custom_embedding_strs=query_bundle.custom_embedding_strs,
            )
        ]
