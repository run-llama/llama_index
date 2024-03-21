# LLM-based reranker implementation

from typing import Callable, List, Optional

# Importing necessary modules and classes from llama_index package
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_CHOICE_SELECT_PROMPT
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import Settings, llm_from_settings_or_context


class LLMRerank(BaseNodePostprocessor):
    """LLM-based reranker."""

    # Class attributes with field descriptions
    top_n: int = Field(description="Top N nodes to return.")
    choice_select_prompt: BasePromptTemplate = Field(
        description="Choice select prompt."
    )
    choice_batch_size: int = Field(description="Batch size for choice select.")
    llm: LLM = Field(description="The LLM to rerank with.")

    # Private attributes
    _format_node_batch_fn: Callable = PrivateAttr()
    _parse_choice_select_answer_fn: Callable = PrivateAttr()

    def __init__(
        self,
        llm: Optional[LLM] = None,
        choice_select_prompt: Optional[BasePromptTemplate] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional[ServiceContext] = None,
        top_n: int = 10,
    ) -> None:
        """Initialize LLMRerank.

        Args:
            llm (Optional[LLM]): The LLM instance to use. Defaults to None.
            choice_select_prompt (Optional[BasePromptTemplate]): The prompt template for choice selection. Defaults to None.
            choice_batch_size (int): The batch size for choice selection. Defaults to 10.
            format_node_batch_fn (Optional[Callable]): Function to format node batch. Defaults to None.
            parse_choice_select_answer_fn (Optional[Callable]): Function to parse choice select answer. Defaults to None.
            service_context (Optional[ServiceContext]): Service context. Defaults to None.
            top_n (int): The number of top-ranked nodes to return. Defaults to 10.

        """
        # Assigning default values if not provided
        choice_select_prompt = choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT
        llm = llm or llm_from_settings_or_context(Settings, service_context)

        # Assigning private attributes
        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )

        # Calling superclass constructor
        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            service_context=service_context,
            top_n=top_n,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"choice_select_prompt": self.choice_select_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "choice_select_prompt" in prompts:
            self.choice_select_prompt = prompts["choice_select_prompt"]

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "LLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Rerank nodes based on LLM predictions.

        Args:
            nodes (List[NodeWithScore]): The list of nodes to rerank.
            query_bundle (Optional[QueryBundle]): The query bundle. Defaults to None.

        Returns:
            List[NodeWithScore]: The reranked nodes.

        Raises:
            ValueError: If query bundle is not provided.

        """
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [
                node.node for node in nodes[idx : idx + self.choice_batch_size]
            ]

            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # Call each batch independently
            raw_response = self.llm.predict(
                self.choice_select_prompt,
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
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]
