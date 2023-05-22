"""Query transform."""

import dataclasses
from abc import abstractmethod
from typing import Dict, Optional, cast

from langchain.input import print_text

from llama_index.indices.query.query_transform.prompts import (
    DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    DEFAULT_IMAGE_OUTPUT_PROMPT,
    DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    DecomposeQueryTransformPrompt,
    ImageOutputQueryTransformPrompt,
    StepDecomposeQueryTransformPrompt,
)
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT
from llama_index.response.schema import Response


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
        query_bundle_or_str: QueryType,
        extra_info: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query transform."""
        extra_info = extra_info or {}
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(
                query_str=query_bundle_or_str,
                custom_embedding_strs=[query_bundle_or_str],
            )
        else:
            query_bundle = query_bundle_or_str

        return self._run(query_bundle, extra_info=extra_info)

    def __call__(
        self,
        query_bundle_or_str: QueryType,
        extra_info: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query processor."""
        return self.run(query_bundle_or_str, extra_info=extra_info)


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
        llm_predictor: Optional[BaseLLMPredictor] = None,
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
        llm_predictor: Optional[BaseLLMPredictor] = None,
        decompose_query_prompt: Optional[DecomposeQueryTransformPrompt] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._decompose_query_prompt = (
            decompose_query_prompt or DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )
        self.verbose = verbose

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        # currently, just get text from the index structure
        index_summary = cast(str, extra_info.get("index_summary", "None"))

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        new_query_str, _ = self._llm_predictor.predict(
            self._decompose_query_prompt,
            query_str=query_str,
            context_str=index_summary,
        )

        if self.verbose:
            print_text(f"> Current query: {query_str}\n", color="yellow")
            print_text(f"> New query: {new_query_str}\n", color="pink")

        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=[new_query_str],
        )


class ImageOutputQueryTransform(BaseQueryTransform):
    """Image output query transform.

    Adds instructions for formatting image output.
    By default, this prompts the LLM to format image output as an HTML <img> tag,
    which can be displayed nicely in jupyter notebook.
    """

    def __init__(
        self,
        width: int = 400,
        query_prompt: Optional[ImageOutputQueryTransformPrompt] = None,
    ) -> None:
        """Init ImageOutputQueryTransform.

        Args:
            width (int): desired image display width in pixels
            query_prompt (ImageOutputQueryTransformPrompt): custom prompt for
                augmenting query with image output instructions.
        """
        self._width = width
        self._query_prompt = query_prompt or DEFAULT_IMAGE_OUTPUT_PROMPT

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        del extra_info  # Unused
        new_query_str = self._query_prompt.format(
            query_str=query_bundle.query_str, image_width=self._width
        )
        new_query_bundle = dataclasses.replace(query_bundle, query_str=new_query_str)
        return new_query_bundle


class StepDecomposeQueryTransform(BaseQueryTransform):
    """Step decompose query transform.

    Decomposes query into a subquery given the current index struct
    and previous reasoning.

    NOTE: doesn't work yet.

    Args:
        llm_predictor (Optional[LLMPredictor]): LLM for generating
            hypothetical documents

    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        step_decompose_query_prompt: Optional[StepDecomposeQueryTransformPrompt] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._step_decompose_query_prompt = (
            step_decompose_query_prompt or DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )
        self.verbose = verbose

    def _run(self, query_bundle: QueryBundle, extra_info: Dict) -> QueryBundle:
        """Run query transform."""
        index_summary = cast(
            str,
            extra_info.get("index_summary", "None"),
        )
        prev_reasoning = cast(Response, extra_info.get("prev_reasoning"))
        fmt_prev_reasoning = f"\n{prev_reasoning}" if prev_reasoning else "None"

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        new_query_str, formatted_prompt = self._llm_predictor.predict(
            self._step_decompose_query_prompt,
            prev_reasoning=fmt_prev_reasoning,
            query_str=query_str,
            context_str=index_summary,
        )
        if self.verbose:
            print_text(f"> Current query: {query_str}\n", color="yellow")
            print_text(f"> Formatted prompt: {formatted_prompt}\n", color="pink")
            print_text(f"> New query: {new_query_str}\n", color="pink")
        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
        )
