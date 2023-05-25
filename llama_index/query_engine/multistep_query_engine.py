from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import Node, NodeWithScore
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.response.schema import RESPONSE_TYPE


def default_stop_fn(stop_dict: Dict) -> bool:
    """Stop function for multi-step query combiner."""
    query_bundle = cast(QueryBundle, stop_dict.get("query_bundle"))
    if query_bundle is None:
        raise ValueError("Response must be provided to stop function.")

    if "none" in query_bundle.query_str.lower():
        return True
    else:
        return False


class MultiStepQueryEngine(BaseQueryEngine):
    """Multi-step query engine.

    This query engine can operate over an existing base query engine,
    along with the multi-step query transform.

    Args:
        query_engine (BaseQueryEngine): A BaseQueryEngine object.
        query_transform (StepDecomposeQueryTransform): A StepDecomposeQueryTransform
            object.
        response_synthesizer (Optional[ResponseSynthesizer]): A ResponseSynthesizer
            object.
        num_steps (Optional[int]): Number of steps to run the multi-step query.
        early_stopping (bool): Whether to stop early if the stop function returns True.
        index_summary (str): A string summary of the index.
        stop_fn (Optional[Callable[[Dict], bool]]): A stop function that takes in a
            dictionary of information and returns a boolean.

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        query_transform: StepDecomposeQueryTransform,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        num_steps: Optional[int] = 3,
        early_stopping: bool = True,
        index_summary: str = "None",
        stop_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> None:
        self._query_engine = query_engine
        self._query_transform = query_transform
        self._response_synthesizer = (
            response_synthesizer
            or ResponseSynthesizer.from_args(
                callback_manager=self._query_engine.callback_manager
            )
        )

        self._index_summary = index_summary
        self._num_steps = num_steps
        self._early_stopping = early_stopping
        # TODO: make interface to stop function better
        self._stop_fn = stop_fn or default_stop_fn
        # num_steps must be provided if early_stopping is False
        if not self._early_stopping and self._num_steps is None:
            raise ValueError("Must specify num_steps if early_stopping is False.")

        callback_manager = self._query_engine.callback_manager
        super().__init__(callback_manager)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        query_event_id = self.callback_manager.on_event_start(CBEventType.QUERY)
        nodes, source_nodes, extra_info = self._query_multistep(query_bundle)

        final_response = self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=source_nodes,
        )
        final_response.extra_info = extra_info

        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=query_event_id)
        return final_response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        event_id = self.callback_manager.on_event_start(CBEventType.QUERY)
        nodes, source_nodes, extra_info = self._query_multistep(query_bundle)

        final_response = await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=source_nodes,
        )
        final_response.extra_info = extra_info

        self.callback_manager.on_event_end(CBEventType.QUERY, event_id=event_id)
        return final_response

    def _combine_queries(
        self, query_bundle: QueryBundle, prev_reasoning: str
    ) -> QueryBundle:
        """Combine queries."""
        transform_extra_info = {
            "prev_reasoning": prev_reasoning,
            "index_summary": self._index_summary,
        }
        query_bundle = self._query_transform(
            query_bundle, extra_info=transform_extra_info
        )
        return query_bundle

    def _query_multistep(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore], Dict[str, Any]]:
        """Run query combiner."""
        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        final_response_extra_info: Dict[str, Any] = {"sub_qa": []}

        text_chunks = []
        source_nodes = []
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

            cur_response = self._query_engine.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {str(cur_response)}"
            )
            text_chunks.append(cur_qa_text)
            for source_node in cur_response.source_nodes:
                source_nodes.append(source_node)
            # update extra info
            final_response_extra_info["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n" f"- {str(cur_response)}\n"
            )
            cur_steps += 1

        nodes = [NodeWithScore(Node(text_chunk)) for text_chunk in text_chunks]
        return nodes, source_nodes, final_response_extra_info
