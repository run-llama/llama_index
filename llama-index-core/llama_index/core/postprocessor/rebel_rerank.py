import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.base import PromptTemplate, PromptType
from llama_index.core.prompts.default_prompts import DEFAULT_REBEL_RERANK_PROMPT
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
    choice_batch_size: int = Field(
        default=10,
        description="Number of nodes to process in each batch for choice selection"
    )

    def __init__(
        self,
        top_n: int = 5,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        rebel_rerank_prompt: Optional[BasePromptTemplate] = None,
        choice_batch_size: int = 10,
    ):
        rebel_rerank_prompt = rebel_rerank_prompt or DEFAULT_REBEL_RERANK_PROMPT
        super().__init__(
            verbose=verbose,
            llm=llm,
            top_n=top_n,
            rebel_rerank_prompt=rebel_rerank_prompt,
            choice_batch_size=choice_batch_size,
        )

    @classmethod
    def class_name(cls) -> str:
        return "REBELRerank"

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"rebel_rerank_prompt": self.rebel_rerank_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "rebel_rerank_prompt" in prompts:
            self.rebel_rerank_prompt = prompts["rebel_rerank_prompt"]

    def _format_node_batch_fn(self, nodes_batch: List[Any]) -> str:
        """Format a batch of nodes into a string for the choice select prompt."""
        fmt_batch_str = ""
        for idx, node in enumerate(nodes_batch):
            fmt_batch_str += f"Choice {idx + 1}:\n{node.get_text()}\n\n"
        return fmt_batch_str

    def _parse_choice_select_answer_fn(
        self, raw_response: str, num_choices: int
    ) -> tuple[List[str], Optional[List[float]]]:
        """Parse the raw response from the LLM into choices and relevance scores."""
        choices = []
        relevances = []
        
        # Clean and normalize the response
        raw_response = raw_response.replace("Doc:", "").replace("Choice:", "")
        parts = [p.strip() for p in raw_response.split(",")]
        
        for part in parts:
            # Handle different formats:
            # "1 (0.8)" or "1(0.8)" -> choice with score
            # "Doc 1" or "Choice 1" -> just choice
            # "1" -> just choice
            part = part.strip()
            
            # Extract the choice number
            choice = None
            score = None
            
            # Try to extract choice and score if in format "X (Y)"
            if "(" in part and ")" in part:
                choice_part = part.split("(")[0].strip()
                try:
                    score = float(part.split("(")[1].split(")")[0])
                    relevances.append(score)
                except (ValueError, IndexError):
                    pass
            else:
                choice_part = part

            # Extract the numeric choice, handling various formats
            import re
            numbers = re.findall(r'\d+', choice_part)
            if numbers:
                choice = numbers[0]  # Take the first number found
                
            if choice is not None:
                choices.append(choice)
            
        if not choices:
            # Fallback: if no valid choices found, use all indices in order
            choices = [str(i+1) for i in range(num_choices)]
        
        if not relevances:
            relevances = None
            
        if self.verbose:
            print_text(f"Parsed choices: {choices}")
            print_text(f"Parsed relevances: {relevances}")
            
        return choices, relevances

    def create_meta_instruction(self, query_str: str) -> List[ChatMessage]:
        """Create the meta instruction for generating the choice select prompt."""
        return [
            ChatMessage(
                role="system",
                content="You are REBEL, an intelligent assistant that can create prompts for ranking passages.",
            ),
            ChatMessage(
                role="user", 
                content=self.rebel_rerank_prompt.format(user_query=query_str)
            ),
        ]

    def run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Run the LLM with the given messages."""
        return self.llm.chat(messages)

    async def arun_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Run the LLM asynchronously with the given messages."""
        return await self.llm.achat(messages)

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

        # Generate the choice select prompt using meta instructions
        meta_messages = self.create_meta_instruction(query_str)
        meta_response = self.run_llm(meta_messages)

        if meta_response.message is None or meta_response.message.content is None:
            raise ValueError("Failed to generate choice select prompt")

        # Create the choice select prompt template
        choice_select_prompt = PromptTemplate(
            template=meta_response.message.content,
            prompt_type=PromptType.CHOICE_SELECT
        )

        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [node.node for node in nodes[idx : idx + self.choice_batch_size]]
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)

            # Call LLM for ranking
            response = self.llm.predict(
                choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                response, len(nodes_batch)
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

            if self.verbose:
                print_text(f"Processed batch {idx//self.choice_batch_size + 1}, "
                          f"found {len(raw_choices)} relevant passages")

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
            : self.top_n
        ]

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Async version of _postprocess_nodes."""
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        query_str = query_bundle.query_str
        meta_messages = self.create_meta_instruction(query_str)
        meta_response = await self.arun_llm(meta_messages)

        if meta_response.message is None or meta_response.message.content is None:
            raise ValueError("Failed to generate choice select prompt")

        # Rest of the implementation similar to _postprocess_nodes but using async calls
        # ...
