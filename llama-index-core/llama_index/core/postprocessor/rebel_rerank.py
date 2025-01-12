import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import REBEL_RERANK_PROMPT
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.utils import print_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_default_llm() -> LLM:
    from llama_index.llms.openai import OpenAI  # pants: no-infer-dep

    return OpenAI(model="gpt-3.5-turbo-16k")


class REBELRerank(BaseNodePostprocessor):
    """REBEL (Rerank Beyond Relevance) reranker."""

    top_n: int = Field(default=5, description="Top N nodes to return from reranking.")
    llm: LLM = Field(
        default_factory=get_default_llm,
        description="LLM to use for REBEL",
    )
    verbose: bool = Field(
        default=False, description="Whether to print intermediate steps."
    )
    rebel_rerank_prompt: SerializeAsAny[BasePromptTemplate] = Field(
        description="REBEL rerank prompt."
    )

    def __init__(
        self,
        top_n: int = 5,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        rebel_rerank_prompt: Optional[BasePromptTemplate] = None,
    ):
        rebel_rerank_prompt = rebel_rerank_prompt or REBEL_RERANK_PROMPT
        super().__init__(
            verbose=verbose,
            llm=llm,
            top_n=top_n,
            rebel_rerank_prompt=rebel_rerank_prompt,
        )

    @classmethod
    def class_name(cls) -> str:
        return "REBELRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        query_str = query_bundle.query_str

        # Replace [USER QUERY] in the meta_prompt with the actual query
        final_meta_prompt = self.meta_prompt.replace("{user_query}", query_str)

        # Call LLM to generate the final choice select prompt
        # Wrap final_meta_prompt in a PromptTemplate
        final_meta_template = PromptTemplate(final_meta_prompt, prompt_type=PromptType.REBEL_RERANK)

        # Now call llm.predict with a BasePromptTemplate
        final_choice_select_prompt_str = self.llm.predict(final_meta_template)

        # Construct the prompt template for choice select
        final_choice_select_prompt = PromptTemplate(
            final_choice_select_prompt_str, prompt_type=PromptType.CHOICE_SELECT
        )
        self.choice_select_prompt = final_choice_select_prompt


        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [node.node for node in nodes[idx : idx + self.choice_batch_size]]

            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response = self.llm.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                raw_response, len(nodes_batch)
            )
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [nodes_batch[i] for i in choice_idxs]
            relevances = relevances or [1.0 for _ in choice_nodes]

            initial_results.extend(
                [
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
            : self.top_n
        ]
