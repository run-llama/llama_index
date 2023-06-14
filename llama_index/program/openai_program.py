from typing import Any, Dict, Generic, Optional, Type

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from llama_index.program.base_program import BasePydanticProgram, Model
from llama_index.prompts.base import Prompt

SUPPORTED_MODEL_NAMES = [
    "gpt-3.5-turbo-0613",
    "gpt-4-0613",
]


def _openai_function(output_cls: Type[Model]) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI function."""
    return {
        "name": "output_pydantic",
        "description": "generate JSON object representing a pydantic model",
        "parameters": output_cls.schema(),
    }


def _openai_function_call() -> Dict[str, Any]:
    """Default OpenAI function to call."""
    return {
        "function_call": {
            "name": "output_pydantic",
        }
    }


class OpenAIPydanticProgram(BasePydanticProgram, Generic[Model]):
    """
    An OpenAI-based function that returns a pydantic model.

    Note: this interface is not yet stable.
    """

    def __init__(
        self,
        output_cls: Type[Model],
        llm: ChatOpenAI,
        prompt: Prompt,
        verbose: bool = False,
    ) -> None:
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[Model],
        prompt_template_str: str,
        llm: Optional[ChatOpenAI] = None,
        verbose: bool = False,
    ) -> "OpenAIPydanticProgram":
        llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("llm must be a ChatOpenAI instance")

        if llm.model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Model name {llm.model_name} not supported. "
                f"Supported model names: {SUPPORTED_MODEL_NAMES}"
            )
        prompt = Prompt(prompt_template_str)
        return cls(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Model:
        formatted_prompt = self._prompt.format(**kwargs)

        ai_message = self._llm.predict_messages(
            messages=[HumanMessage(content=formatted_prompt)],
            functions=[_openai_function(self._output_cls)],
            # TODO: support forcing the desired function call
            # function_call=_openai_function_call(),
        )
        if "function_call" not in ai_message.additional_kwargs:
            raise ValueError(
                "Expected function call in ai_message.additional_kwargs, "
                "but none found."
            )

        function_call = ai_message.additional_kwargs["function_call"]
        name = function_call["name"]
        arguments_str = function_call["arguments"]
        if self._verbose:
            print(f"Function call: {name} with args: {arguments_str}")

        output = self.output_cls.parse_raw(function_call["arguments"])
        return output
