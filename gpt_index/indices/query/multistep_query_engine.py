from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast
from gpt_index.data_structs.node_v2 import Node, NodeWithScore
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.indices.response.response_synthesis import ResponseSynthesizer
from gpt_index.response.schema import RESPONSE_TYPE


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
    def __init__(
        self,
        query_engine: BaseQueryEngine,
        query_transform: StepDecomposeQueryTransform,
        response_synthesizer: Optional[ResponseSynthesizer] = None,
        num_steps: Optional[int] = 3,
        early_stopping: bool = True,
        stop_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> None:
        self._query_engine = query_engine
        self._query_transform = query_transform
        self._response_synthesizer = (
            response_synthesizer or ResponseSynthesizer.from_args()
        )

        self._num_steps = num_steps
        self._early_stopping = early_stopping
        # TODO: make interface to stop function better
        self._stop_fn = stop_fn or default_stop_fn
        # num_steps must be provided if early_stopping is False
        if not self._early_stopping and self._num_steps is None:
            raise ValueError("Must specify num_steps if early_stopping is False.")

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes, source_nodes, extra_info = self._query_multistep(query_bundle)
        return self.synthesize(query_bundle, nodes, source_nodes, extra_info)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes, source_nodes, extra_info = self._query_multistep(query_bundle)
        return await self.synthesize(query_bundle, nodes, source_nodes, extra_info)

    def _combine_queries(
        self, query_bundle: QueryBundle, prev_reasoning: str
    ) -> QueryBundle:
        """Combine queries."""
        transform_extra_info = {
            "prev_reasoning": prev_reasoning,
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

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: Sequence[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        extra_info: Optional[dict] = None,
    ) -> RESPONSE_TYPE:
        final_response = self._response_synthesizer.synthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )
        final_response.extra_info = extra_info
        return final_response

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: Sequence[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        extra_info: Optional[dict] = None,
    ) -> RESPONSE_TYPE:
        final_response = await self._response_synthesizer.asynthesize(
            query_bundle=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )
        final_response.extra_info = extra_info
        return final_response
