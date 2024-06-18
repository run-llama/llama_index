# Reference: https://github.com/togethercomputer/MoA

import logging
from typing import Any, Dict, List
import copy
import asyncio
import sys

from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class MixtureOfAgentsPack(BaseLlamaPack):
    def __init__(
        self,
        llm: LLM,
        reference_llms: List[LLM],
        num_layers: int = 3,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self.llm = llm
        self.reference_llms = reference_llms
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        self.temperature = temperature

    def inject_references_to_messages(
        self,
        messages: List[ChatMessage],
        references: List[str],
    ) -> List[ChatMessage]:
        messages = copy.deepcopy(messages)

        system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

    Responses from models:"""

        for i, reference in enumerate(references):
            system += f"\n{i+1}. {reference}"

        if messages[0].role == "system":
            messages[0].content += "\n\n" + system

        else:
            messages = [ChatMessage(role="system", content=system), *messages]

        return messages

    async def agenerate_with_references(
        self,
        llm: LLM,
        messages: List[ChatMessage],
        references: List[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        if len(references) > 0:
            messages = self.inject_references_to_messages(messages, references)

        return str(
            await llm.achat(messages, max_tokens=max_tokens, temperature=temperature)
        ).strip()

    async def get_answer(self, query_str: str) -> str:
        messages = []

        messages.append(ChatMessage(role="user", content=query_str))

        references = []

        if len(self.reference_llms) > 0:
            prev_references = []

            for layer in range(self.num_layers):
                logger.info(
                    f"Round {layer+1}/{self.num_layers} to collecting reference responses."
                )

                references = []

                jobs = [
                    self.agenerate_with_references(
                        llm=reference_llm,
                        messages=messages,
                        references=prev_references,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    for reference_llm in self.reference_llms
                ]

                references = await asyncio.gather(*jobs)

                if layer < self.num_layers - 1:
                    prev_references = references

                    references = []

        return await self.agenerate_with_references(
            llm=self.llm,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            references=references,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "llm": self.llm,
            "reference_llms": self.reference_llms,
            "num_layers": self.num_layers,
        }

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return asyncio.run(self.get_answer(query_str))
