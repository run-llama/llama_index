import asyncio
import json
from dataclasses import dataclass
from typing import Union

from pydantic import BaseModel

from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.decorators import step, service
from llama_index.core.prompts import PromptTemplate

# pip install llama-index-llms-ollama
from llama_index.llms.ollama import Ollama


class Car(BaseModel):
    brand: str
    model: str
    power: int


class CarCollection(BaseModel):
    cars: list[Car]


@dataclass
class ExtractionDone(Event):
    output: str
    passage: str


@dataclass
class ValidationErrorEvent(Event):
    error: str
    wrong_output: str
    passage: str


EXTRACTION_PROMPT = PromptTemplate(
    """
Context information is below:
---------------------
{passage}
---------------------

Given the context information and not prior knowledge, create a JSON object from the information in the context.
The JSON object must follow the JSON schema:
{schema}

"""
)

REFLECTION_PROMPT = PromptTemplate(
    """
You already created this output previously:
---------------------
{wrong_answer}
---------------------

This caused the JSON decode error: {error}

Try again, the response must contain only valid JSON code. Do not add any sentence before or after the JSON object.
Do not repeat the schema.
"""
)


class ReflectionWorkflow(Workflow):
    @service()
    async def run_ollama(self, prompt: str) -> str:
        llm = Ollama(model="llama3", request_timeout=30.0)
        output = await llm.acomplete(prompt)
        return str(output)

    @step()
    async def extract(
        self, ev: Union[StartEvent, ValidationErrorEvent]
    ) -> Union[StopEvent, ExtractionDone]:
        if isinstance(ev, StartEvent):
            passage = ev.get("passage")
            if not passage:
                return StopEvent(msg="Please provide some text in input")
            reflection_prompt = ""
        elif isinstance(ev, ValidationErrorEvent):
            passage = ev.passage
            reflection_prompt = REFLECTION_PROMPT.format(
                wrong_answer=ev.wrong_output, error=ev.error
            )

        prompt = EXTRACTION_PROMPT.format(
            passage=passage, schema=CarCollection.schema_json()
        )
        if reflection_prompt:
            prompt += reflection_prompt

        output = await self.run_ollama(prompt)

        return ExtractionDone(output=str(output), passage=passage)

    @step()
    async def validate(
        self, ev: ExtractionDone
    ) -> Union[StopEvent, ValidationErrorEvent]:
        try:
            json.loads(ev.output)
        except Exception as e:
            print("Validation failed, retrying...")
            return ValidationErrorEvent(
                error=str(e), wrong_output=ev.output, passage=ev.passage
            )

        return StopEvent(msg=ev.output)


async def main():
    w = ReflectionWorkflow(timeout=120, verbose=False)

    ret = await w.run(
        passage="I own two cars: a Fiat Panda with 45Hp and a Honda Civic with 330Hp."
    )
    print(ret)
    print(f"Average time to run Ollama: {w.run_ollama.avg_time:.2f}s")

    w.draw_all_possible_flows()
    w.draw_most_recent_execution()


if __name__ == "__main__":
    asyncio.run(main())
