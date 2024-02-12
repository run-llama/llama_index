"""Query transform."""

import dataclasses
from abc import abstractmethod
from typing import Any, Dict, Optional, cast

from llama_index.bridge.pydantic import Field
from llama_index.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.response.schema import Response
from llama_index.indices.query.query_transform.prompts import (
    DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    DEFAULT_IMAGE_OUTPUT_PROMPT,
    DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT,
    DecomposeQueryTransformPrompt,
    ImageOutputQueryTransformPrompt,
    StepDecomposeQueryTransformPrompt,
)
from llama_index.llm_predictor.base import LLMPredictorType
from llama_index.llms.utils import resolve_llm
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT
from llama_index.prompts.mixin import PromptDictType, PromptMixin, PromptMixinType
from llama_index.schema import QueryBundle, QueryType
from llama_index.utils import print_text


class BaseQueryTransform(ChainableMixin, PromptMixin):
    """Base class for query transform.

    A query transform augments a raw query string with associated transformations
    to improve index querying.

    The query transformation is performed before the query is sent to the index.

    """

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        # TODO: keep this for now since response synthesizers don't generally have sub-modules
        return {}

    @abstractmethod
    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""

    def run(
        self,
        query_bundle_or_str: QueryType,
        metadata: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query transform."""
        metadata = metadata or {}
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(
                query_str=query_bundle_or_str,
                custom_embedding_strs=[query_bundle_or_str],
            )
        else:
            query_bundle = query_bundle_or_str

        return self._run(query_bundle, metadata=metadata)

    def __call__(
        self,
        query_bundle_or_str: QueryType,
        metadata: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query processor."""
        return self.run(query_bundle_or_str, metadata=metadata)

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return QueryTransformComponent(query_transform=self)


class IdentityQueryTransform(BaseQueryTransform):
    """Identity query transform.

    Do nothing to the query.

    """

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
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
        llm: Optional[LLMPredictorType] = None,
        hyde_prompt: Optional[BasePromptTemplate] = None,
        include_original: bool = True,
    ) -> None:
        """Initialize HyDEQueryTransform.

        Args:
            llm_predictor (Optional[LLM]): LLM for generating
                hypothetical documents
            hyde_prompt (Optional[BasePromptTemplate]): Custom prompt for HyDE
            include_original (bool): Whether to include original query
                string as one of the embedding strings
        """
        super().__init__()

        self._llm = llm or resolve_llm("default")
        self._hyde_prompt = hyde_prompt or DEFAULT_HYDE_PROMPT
        self._include_original = include_original

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"hyde_prompt": self._hyde_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "hyde_prompt" in prompts:
            self._hyde_prompt = prompts["hyde_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        # TODO: support generating multiple hypothetical docs
        query_str = query_bundle.query_str
        hypothetical_doc = self._llm.predict(self._hyde_prompt, context_str=query_str)
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
        llm_predictor (Optional[LLM]): LLM for generating
            hypothetical documents

    """

    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        decompose_query_prompt: Optional[DecomposeQueryTransformPrompt] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm = llm or resolve_llm("default")
        self._decompose_query_prompt = (
            decompose_query_prompt or DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )
        self.verbose = verbose

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"decompose_query_prompt": self._decompose_query_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "decompose_query_prompt" in prompts:
            self._decompose_query_prompt = prompts["decompose_query_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        # currently, just get text from the index structure
        index_summary = cast(str, metadata.get("index_summary", "None"))

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        new_query_str = self._llm.predict(
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

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"query_prompt": self._query_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "query_prompt" in prompts:
            self._query_prompt = prompts["query_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        del metadata  # Unused
        new_query_str = self._query_prompt.format(
            query_str=query_bundle.query_str, image_width=self._width
        )
        return dataclasses.replace(query_bundle, query_str=new_query_str)


class StepDecomposeQueryTransform(BaseQueryTransform):
    """Step decompose query transform.

    Decomposes query into a subquery given the current index struct
    and previous reasoning.

    NOTE: doesn't work yet.

    Args:
        llm_predictor (Optional[LLM]): LLM for generating
            hypothetical documents

    """

    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        step_decompose_query_prompt: Optional[StepDecomposeQueryTransformPrompt] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        super().__init__()
        self._llm = llm or resolve_llm("default")
        self._step_decompose_query_prompt = (
            step_decompose_query_prompt or DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT
        )
        self.verbose = verbose

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"step_decompose_query_prompt": self._step_decompose_query_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "step_decompose_query_prompt" in prompts:
            self._step_decompose_query_prompt = prompts["step_decompose_query_prompt"]

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        index_summary = cast(
            str,
            metadata.get("index_summary", "None"),
        )
        prev_reasoning = cast(Response, metadata.get("prev_reasoning"))
        fmt_prev_reasoning = f"\n{prev_reasoning}" if prev_reasoning else "None"

        # given the text from the index, we can use the query bundle to generate
        # a new query bundle
        query_str = query_bundle.query_str
        new_query_str = self._llm.predict(
            self._step_decompose_query_prompt,
            prev_reasoning=fmt_prev_reasoning,
            query_str=query_str,
            context_str=index_summary,
        )
        if self.verbose:
            print_text(f"> Current query: {query_str}\n", color="yellow")
            print_text(f"> New query: {new_query_str}\n", color="pink")
        return QueryBundle(
            query_str=new_query_str,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
        )


class QueryTransformComponent(QueryComponent):
    """Query transform component."""

    query_transform: BaseQueryTransform = Field(..., description="Query transform.")

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        # TODO: not implemented yet

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        if "query_str" not in input:
            raise ValueError("Input must have key 'query_str'")
        input["query_str"] = validate_and_convert_stringable(input["query_str"])

        input["metadata"] = input.get("metadata", {})

        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self._query_transform.run(
            kwargs["query_str"],
            metadata=kwargs["metadata"],
        )
        return {"query_str": output.query_str}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component."""
        # TODO: true async not implemented yet
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"query_str"}, optional_keys={"metadata"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"query_str"})
