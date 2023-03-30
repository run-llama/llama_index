"""Query combiner class."""

from abc import abstractmethod
from typing import Any, Callable, Dict, Generator, Optional, cast

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct as IndexStruct
from gpt_index.indices.query.query_transform.base import (
    BaseQueryTransform,
    StepDecomposeQueryTransform,
)
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.builder import ResponseBuilder, ResponseMode, TextChunk
from gpt_index.indices.service_context import ServiceContext
from gpt_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from gpt_index.response.schema import RESPONSE_TYPE, Response


class BaseQueryCombiner:
    """Base query combiner."""

    def __init__(
        self,
        index_struct: IndexStruct,
        query_transform: BaseQueryTransform,
        query_runner: Any,  # NOTE: type as Any to avoid circular dependency
    ) -> None:
        """Init params."""
        self._index_struct = index_struct
        self._query_transform = query_transform
        from gpt_index.indices.query.query_runner import QueryRunner

        assert isinstance(query_runner, QueryRunner)
        self._query_runner = query_runner

    @abstractmethod
    def run(self, query_bundle: QueryBundle, level: int = 0) -> RESPONSE_TYPE:
        """Run query combiner."""

    async def arun(self, query_bundle: QueryBundle, level: int = 0) -> RESPONSE_TYPE:
        """Async run query combiner."""
        return self.run(query_bundle, level=level)


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

    def run(self, query_bundle: QueryBundle, level: int = 0) -> RESPONSE_TYPE:
        """Run query combiner."""
        updated_query_bundle = self._prepare_update(query_bundle)
        return self._query_runner.query_transformed(
            updated_query_bundle, self._index_struct, level=level
        )

    async def arun(self, query_bundle: QueryBundle, level: int = 0) -> RESPONSE_TYPE:
        """Run query combiner."""
        updated_query_bundle = self._prepare_update(query_bundle)
        return await self._query_runner.aquery_transformed(
            updated_query_bundle, self._index_struct, level=level
        )


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
        query_runner: Any,
        service_context: Optional[ServiceContext] = None,
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
        super().__init__(
            index_struct, query_transform=query_transform, query_runner=query_runner
        )
        self._index_struct = index_struct
        self._query_transform = query_transform
        from gpt_index.indices.query.query_runner import QueryRunner

        assert isinstance(query_runner, QueryRunner)
        self._query_runner = query_runner
        self._service_context = service_context or ServiceContext.from_defaults()
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
            self._service_context,
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

    def run(self, query_bundle: QueryBundle, level: int = 0) -> RESPONSE_TYPE:
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

            cur_response = self._query_runner.query_transformed(
                updated_query_bundle, self._index_struct, level=level
            )

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {str(cur_response)}"
            )
            self.response_builder.add_text_chunks([TextChunk(cur_qa_text)])
            for source_node in cur_response.source_nodes:
                self.response_builder.add_node_with_score(source_node)
            # update extra info
            final_response_extra_info["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n" f"- {str(cur_response)}\n"
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
    query_runner: Any,  # NOTE: type as Any to avoid circular dependency
    extra_kwargs: Optional[Dict] = None,
) -> BaseQueryCombiner:
    """Get default query combiner."""
    extra_kwargs = extra_kwargs or {}
    if isinstance(query_transform, StepDecomposeQueryTransform):
        return MultiStepQueryCombiner(
            index_struct,
            query_transform=query_transform,
            query_runner=query_runner,
            service_context=extra_kwargs.get("service_context", None),
        )
    else:
        return SingleQueryCombiner(
            index_struct, query_transform=query_transform, query_runner=query_runner
        )
