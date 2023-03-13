"""Query combiner class."""

from abc import ABC, abstractmethod
from typing import List, Optional, cast, Dict, Callable
from gpt_index.response.schema import Response
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.indices.query.query_transform.base import (
    BaseQueryTransform,
    StepDecomposeQueryTransform,
)
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor


class BaseQueryCombiner:
    """Base query combiner."""

    def __init__(
        self,
        index_struct: IndexStruct,
        query_transform: Optional[BaseQueryTransform] = None,
    ) -> None:
        """Init params."""
        self._index_struct = index_struct
        self._query_transform = query_transform

    @abstractmethod
    def run(self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle) -> Response:
        """Run query combiner."""


class SingleQueryCombiner(BaseQueryCombiner):
    """Single query combiner.

    Only runs for a single query. Invalid for multiple queries.

    """

    def run(self, query_obj: BaseGPTIndexQuery, query_bundle: QueryBundle) -> Response:
        """Run query combiner."""
        transform_extra_info = {
            "index_struct": self._index_struct,
        }
        updated_query_bundle = self._query_transform(
            query_bundle, extra_info=transform_extra_info
        )
        return query_obj.query(updated_query_bundle)


def default_stop_fn(stop_dict: Dict) -> bool:
    """Default stop function for multi-step query combiner."""
    query_bundle = cast(QueryBundle, stop_dict.get(["query_bundle"]))
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
        query_transform: Optional[BaseQueryTransform] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        num_steps: Optional[int] = 5,
        early_stopping: bool = True,
        stop_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> None:
        """Init params."""
        super().__init__(index_struct, query_transform=query_transform)
        self._index_struct = index_struct
        self._query_transform = query_transform
        self._llm_predictor = llm_predictor or LLMPredictor()
        self._num_steps = num_steps
        self._early_stopping = early_stopping
        # TODO: make interface to stop function better
        self._stop_fn = stop_fn or default_stop_fn
        # num_steps must be provided if early_stopping is False
        if not self._early_stopping and self._num_steps is None:
            raise ValueError("Must specify num_steps if early_stopping is False.")

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
        while not should_stop:
            if cur_steps >= self._num_steps:
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
            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n" f"- {cur_response.response}\n"
            )
            cur_steps += 1

        return cast(Response, cur_response)


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
