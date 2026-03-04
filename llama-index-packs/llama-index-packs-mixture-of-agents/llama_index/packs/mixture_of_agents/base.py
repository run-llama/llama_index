# Reference: https://github.com/togethercomputer/MoA

import logging
from typing import Any, Dict, List, Union
import copy
import sys

from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.llms import ChatMessage
from llama_index.core.async_utils import asyncio_run

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.workflow import (
    Workflow,
    Event,
    step,
    Context,
    StartEvent,
    StopEvent,
)


class GenerateEvent(Event):
    llm: LLM
    messages: List[ChatMessage]
    references: List[str]


class GenerateResultEvent(Event):
    result: str


class LoopEvent(Event):
    pass


class GatherEvent(Event):
    pass


class MixtureOfAgentWorkflow(Workflow):
    def __init__(
        self,
        main_llm: LLM,
        reference_llms: List[LLM],
        num_layers: int = 3,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.reference_llms = reference_llms
        self.main_llm = main_llm
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
            system += f"\n{i + 1}. {reference}"

        if messages[0].role == "system":
            messages[0].content += "\n\n" + system

        else:
            messages = [ChatMessage(role="system", content=system), *messages]

        return messages

    @step(pass_context=True)
    async def get_answer(
        self, ctx: Context, ev: Union[StartEvent, LoopEvent]
    ) -> Union[GatherEvent, GenerateEvent, StopEvent]:
        # this is a for loop for num_layers
        # in every loop, we need to call agenerate_with_references with every llm
        # then we need to use the result of every llm to update the references
        # and then we need a final call to get the answer

        if isinstance(ev, StartEvent):
            ctx.data["prev_references"] = []
            ctx.data["current_layer"] = 0
            ctx.data["query_str"] = ev.get("query_str")
            ctx.data["messages"] = [
                ChatMessage(role="user", content=ev.get("query_str"))
            ]

        current_layer = ctx.data.get("current_layer", 0)
        if current_layer >= self.num_layers:
            # do final call, return stop event
            final_result = str(
                await self.main_llm.achat(
                    ctx.data["messages"],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            ).strip()
            return StopEvent(result=final_result)
        else:
            # send a generate event to every reference llm
            for llm in self.reference_llms:
                # send generate event
                self.send_event(
                    GenerateEvent(
                        llm=llm,
                        messages=ctx.data["messages"],
                        references=ctx.data["prev_references"],
                    )
                )

            return GatherEvent()

    @step(pass_context=True)
    async def agenerate_with_references(
        self, ctx: Context, ev: GenerateEvent
    ) -> GenerateResultEvent:
        messages = ev.messages
        if len(ev.references) > 0:
            messages = self.inject_references_to_messages(ev.messages, ev.references)
        result = str(
            await ev.llm.achat(
                messages, max_tokens=self.max_tokens, temperature=self.temperature
            )
        ).strip()
        return GenerateResultEvent(result=result)

    @step(pass_context=True)
    async def gather_results(
        self, ctx: Context, ev: Union[GatherEvent, GenerateResultEvent]
    ) -> Union[LoopEvent, None]:
        events = ctx.collect_events(
            ev, [GenerateResultEvent] * len(self.reference_llms)
        )
        if not events:
            return None

        ctx.data["current_layer"] += 1
        ctx.data["prev_references"] = [ev.result for ev in events]

        return LoopEvent()


class MixtureOfAgentsPack(BaseLlamaPack):
    def __init__(
        self,
        llm: LLM,
        reference_llms: List[LLM],
        num_layers: int = 3,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 200,
    ) -> None:
        self._wf = MixtureOfAgentWorkflow(
            llm, reference_llms, num_layers, max_tokens, temperature, timeout=timeout
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "llm": self._wf.main_llm,
            "reference_llms": self._wf.reference_llms,
            "num_layers": self._wf.num_layers,
            "temperature": self._wf.temperature,
            "max_tokens": self._wf.max_tokens,
        }

    def run(self, query_str: str, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return asyncio_run(self._wf.run(query_str=query_str))
