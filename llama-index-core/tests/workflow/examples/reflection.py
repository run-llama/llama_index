import asyncio
import json
from dataclasses import dataclass
from typing import Union

from pydantic import BaseModel

from llama_index.core import set_global_handler

set_global_handler("arize_phoenix")

from llama_index.core.instrumentation import get_dispatcher

dispatcher = get_dispatcher("my_app")

from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

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
    @step()
    async def extract(
        self, ev: Union[StartEvent, ValidationErrorEvent]
    ) -> Union[StopEvent, ExtractionDone]:
        if isinstance(ev, StartEvent):
            passage = ev.get("passage")
            if not passage:
                return StopEvent(result="Please provide some text in input")
            reflection_prompt = ""
        elif isinstance(ev, ValidationErrorEvent):
            passage = ev.passage
            reflection_prompt = REFLECTION_PROMPT.format(
                wrong_answer=ev.wrong_output, error=ev.error
            )

        llm = Ollama(model="llama3", request_timeout=30)
        prompt = EXTRACTION_PROMPT.format(
            passage=passage, schema=CarCollection.schema_json()
        )
        if reflection_prompt:
            prompt += reflection_prompt

        output = llm.complete(prompt)

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

        return StopEvent(result=ev.output)


async def main():
    w = ReflectionWorkflow(timeout=60, verbose=True)

    ret = await w.run(
        passage="I own two cars: a Fiat Panda with 45Hp and a Honda Civic with 330Hp."
    )
    print(ret)


if __name__ == "__main__":
    asyncio.run(main())
