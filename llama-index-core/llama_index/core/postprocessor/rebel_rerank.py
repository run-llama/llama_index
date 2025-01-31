import logging
from typing import Any, Dict, List, Optional, Sequence, Callable, Union
import re

from llama_index.core.bridge.pydantic import Field, PrivateAttr, SerializeAsAny
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.base import PromptTemplate, PromptType
from llama_index.core.prompts.default_prompts import DEFAULT_REBEL_META_PROMPT
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.utils import print_text
from llama_index.core.settings import Settings
from llama_index.core.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_default_llm() -> LLM:
    from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
    return OpenAI(model="gpt-3.5-turbo-16k")


class REBELRerank(BaseNodePostprocessor):
    """REBEL (Rerank Beyond Relevance) reranker."""

    top_n: int = Field(description="Top N nodes to return.")
    rebel_prompt: SerializeAsAny[BasePromptTemplate] = Field(
        description="REBEL prompt that generates ranking instructions."
    )
    choice_batch_size: int = Field(description="Batch size for choice select.")
    llm: LLM = Field(description="The LLM to rerank with.")
    verbose: bool = Field(
        default=False, description="Whether to print intermediate steps."
    )
    choice_select_prompt: Optional[SerializeAsAny[BasePromptTemplate]] = Field(
        default=None, description="Generated prompt for choice selection."
    )

    _format_node_batch_fn: Callable = PrivateAttr()
    _parse_choice_select_answer_fn: Callable = PrivateAttr()

    def __init__(
        self,
        llm: Optional[LLM] = None,
        rebel_prompt: Optional[BasePromptTemplate] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        top_n: int = 10,
    ) -> None:
        """Initialize params."""
        rebel_prompt = rebel_prompt or DEFAULT_REBEL_META_PROMPT
        llm = llm or Settings.llm

        super().__init__(
            llm=llm,
            rebel_prompt=rebel_prompt,
            choice_batch_size=choice_batch_size,
            top_n=top_n,
        )
        
        self._format_node_batch_fn = format_node_batch_fn or default_format_node_batch_fn
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )

    @classmethod
    def class_name(cls) -> str:
        return "REBELRerank"

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"rebel_prompt": self.rebel_prompt, "choice_select_prompt": self.choice_select_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "rebel_prompt" in prompts:
            self.rebel_prompt = prompts["rebel_prompt"]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        query_str = query_bundle.query_str
        
        ranking_prompt = self.llm.predict(
            self.rebel_prompt,
            user_query=query_str,
        )

        self.choice_select_prompt = PromptTemplate(
            ranking_prompt,
            prompt_type=PromptType.CHOICE_SELECT,
        )
        
        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [node.node for node in nodes[idx : idx + self.choice_batch_size]]
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            
            response = self.llm.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str
            )
            
            raw_choices, relevances = self._parse_choice_select_answer_fn(
                response, len(nodes_batch)
            )

            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [nodes_batch[i] for i in choice_idxs]
            # relevances = relevances or [1.0 for _ in choice_nodes]
            
            initial_results.extend(
                [
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        final_results = sorted(
            initial_results, 
            key=lambda x: x.score or 0.0, 
            reverse=True
        )[:self.top_n]

        return final_results

    # def run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
    #     """Run the LLM with the given messages."""
    #     return self.llm.chat(messages)

    # async def arun_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
    #     """Run the LLM asynchronously with the given messages."""
    #     return await self.llm.achat(messages)
