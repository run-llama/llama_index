"""LLM reranker."""
from typing import Callable, List, Optional

from llama_index.data_structs.node import NodeWithScore
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from llama_index.prompts.choice_select import DEFAULT_CHOICE_SELECT_PROMPT
from llama_index.prompts.prompts import QuestionAnswerPrompt


class LLMRerank(BaseNodePostprocessor):
    """LLM-based reranker."""

    def __init__(
        self,
        choice_select_prompt: Optional[QuestionAnswerPrompt] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional[ServiceContext] = None,
        top_n: int = 10,
    ) -> None:
        self._choice_select_prompt = (
            choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT
        )
        self._choice_batch_size = choice_batch_size
        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )
        self._service_context = service_context or ServiceContext.from_defaults()
        self._top_n = top_n

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self._choice_batch_size):
            nodes_batch = [
                node.node for node in nodes[idx : idx + self._choice_batch_size]
            ]

            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response, _ = self._service_context.llm_predictor.predict(
                self._choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                raw_response, len(nodes_batch)
            )
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
            relevances = relevances or [1.0 for _ in choice_nodes]
            initial_results.extend(
                [
                    NodeWithScore(node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        results = sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
            : self._top_n
        ]
        return results
