"""Query combiner class."""

from abc import abstractmethod
from typing import Any, Callable, Dict, Generator, Optional, cast

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.query_transform.base import (
    BaseQueryTransform,
    StepDecomposeQueryTransform,
)
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseBuilder, ResponseMode, TextChunk
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.response.schema import Response


class BaseQueryCombiner:
    """Base query combiner."""

    def __init__(
        self,
        index_struct: IndexStruct,
        query_transform: BaseQueryTransform,
    ) -> None:
        """Init params."""
        self._index_struct = index_struct
        self._query_transform = query_transform

    @abstractmethod
    def run(self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle) -> Response:
        """Run query combiner."""

    async def arun(
        self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle
    ) -> Response:
        """Async run query combiner."""
        return self.run(query_obj, query_bundle)


class SingleQueryCombiner(BaseQueryCombiner):
    """Single query combiner.

    Only runs for a single query. Invalid for multiple queries.

    """

    def _prepare_update(self, query_bundle: QueryBundle) -> QueryBundle:
        """Prepare update."""
        transform_extra_info = {
            "index_struct": self._index_struct,
        }
        updated_query_bundle = self._query_transform(
            query_bundle, extra_info=transform_extra_info
        )
        return updated_query_bundle

    def run(self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle) -> Response:
        """Run query combiner."""
        updated_query_bundle = self._prepare_update(query_bundle)
        return query_obj.query(updated_query_bundle)

    async def arun(
        self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle
    ) -> Response:
        """Run query combiner."""
        updated_query_bundle = self._prepare_update(query_bundle)
        return await query_obj.aquery(updated_query_bundle)


def default_stop_fn(stop_dict: Dict) -> bool:
    """Stop function for multi-step query combiner."""
    query_bundle = cast(QueryBundle, stop_dict.get("query_bundle"))
    if query_bundle is None:
        raise ValueError("Response must be provided to stop function.")

    if "none" in query_bundle.query_str.lower():
        return True
    else:
        return False


class MultiStepQueryCombiner(BaseQueryCombiner):
    """Multi-step query combiner.

    Runs over queries in succession.

    """

    def __init__(
        self,
        index_struct: IndexStruct,
        query_transform: BaseQueryTransform,
        llm_predictor: Optional[LLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        refine_template: Optional[RefinePrompt] = None,
        response_mode: ResponseMode = ResponseMode.DEFAULT,
        response_kwargs: Optional[Dict] = None,
        num_steps: Optional[int] = 3,
        early_stopping: bool = True,
        stop_fn: Optional[Callable[[Dict], bool]] = None,
        use_async: bool = True,
    ) -> None:
        """Init params."""
        super().__init__(index_struct, query_transform=query_transform)
        self._index_struct = index_struct
        self._query_transform = query_transform
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            self._llm_predictor, chunk_size_limit=None
        )
        self._num_steps = num_steps
        self._early_stopping = early_stopping
        # TODO: make interface to stop function better
        self._stop_fn = stop_fn or default_stop_fn
        # num_steps must be provided if early_stopping is False
        if not self._early_stopping and self._num_steps is None:
            raise ValueError("Must specify num_steps if early_stopping is False.")

        self._response_mode = ResponseMode(response_mode)
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
        self.response_builder = ResponseBuilder(
            self._prompt_helper,
            self._llm_predictor,
            self.text_qa_template,
            self.refine_template,
            use_async=use_async,
        )
        self._response_kwargs = response_kwargs or {}

    def _combine_queries(
        self, query_bundle: QueryBundle, prev_reasoning: str
    ) -> QueryBundle:
        """Combine queries."""
        transform_extra_info = {
            "index_struct": self._index_struct,
            "prev_reasoning": prev_reasoning,
        }
        query_bundle = self._query_transform(
            query_bundle, extra_info=transform_extra_info
        )
        return query_bundle

    def run(self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle) -> Response:
        """Run query combiner."""
        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        self.response_builder.reset()
        final_response_extra_info: Dict[str, Any] = {"sub_qa": []}

        while not should_stop:
            if self._num_steps is not None and cur_steps >= self._num_steps:
                should_stop = True
                break
            elif should_stop:
                break

            updated_query_bundle = self._combine_queries(query_bundle, prev_reasoning)

            # TODO: make stop logic better
            stop_dict = {"query_bundle": updated_query_bundle}
            if self._stop_fn(stop_dict):
                should_stop = True
                break

            cur_response = query_obj.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {cur_response.response}"
            )
            self.response_builder.add_text_chunks([TextChunk(cur_qa_text)])
            for source_node in cur_response.source_nodes:
                self.response_builder.add_source_node(source_node)
            # update extra info
            final_response_extra_info["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n" f"- {cur_response.response}\n"
            )
            cur_steps += 1

        # synthesize a final response
        final_response_str = self.response_builder.get_response(
            query_bundle.query_str,
            mode=self._response_mode,
            **self._response_kwargs,
        )
        if isinstance(final_response_str, Generator):
            raise ValueError("Currently streaming is not supported for query combiner.")
        return Response(
            response=final_response_str,
            source_nodes=self.response_builder.get_sources(),
            extra_info=final_response_extra_info,
        )


def get_default_query_combiner(
    index_struct: IndexStruct,
    query_transform: BaseQueryTransform,
    extra_kwargs: Optional[Dict] = None,
) -> BaseQueryCombiner:
    """Get default query combiner."""
    extra_kwargs = extra_kwargs or {}
    if isinstance(query_transform, StepDecomposeQueryTransform):
        return MultiStepQueryCombiner(
            index_struct,
            query_transform=query_transform,
            llm_predictor=extra_kwargs.get("llm_predictor", None),
        )
    else:
        return SingleQueryCombiner(index_struct, query_transform=query_transform)
