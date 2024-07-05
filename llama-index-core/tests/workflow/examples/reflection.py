import asyncio
import json
from dataclasses import dataclass

from pydantic import BaseModel

from llama_index.core.workflow.events import Event, StartEvent, StopEvent
from llama_index.core.workflow.workflow import Workflow
from llama_index.core.workflow.decorators import step
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


@dataclass
class ValidationErrorEvent(Event):
    error: str
    wrong_output: str


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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Shared state: to be refactored into a better workflow context manager
        self.passage: str | None = None

    @step(StartEvent, ValidationErrorEvent)
    async def extract(self, ev: StartEvent | ValidationErrorEvent):
        if isinstance(ev, StartEvent):
            self.passage = ev.get("passage")
            if not self.passage:
                return StopEvent(msg="Please provide some text in input")
            reflection_prompt = ""
        else:
            reflection_prompt = REFLECTION_PROMPT.format(
                wrong_answer=ev.wrong_output, error=ev.error
            )

        llm = Ollama(model="llama3", request_timeout=30)
        prompt = EXTRACTION_PROMPT.format(
            passage=self.passage, schema=CarCollection.schema_json()
        )
        if reflection_prompt:
            prompt += reflection_prompt

        output = llm.complete(prompt)

        return ExtractionDone(output=str(output))

    @step(ExtractionDone)
    async def validate(self, ev: ExtractionDone):
        try:
            json.loads(ev.output)
        except Exception as e:
            print("Validation failed, retrying...")
            return ValidationErrorEvent(error=str(e), wrong_output=ev.output)

        return StopEvent(msg=ev.output)


async def main():
    w = ReflectionWorkflow(timeout=120)

    ret = await w.run(
        passage="I own two cars: a Fiat Panda with 45Hp and a Honda Civic with 330Hp."
    )
    print(ret)


if __name__ == "__main__":
    asyncio.run(main())
