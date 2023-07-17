from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import is_function_calling_model, to_openai_function
from llama_index.program.llm_prompt_program import BaseLLMFunctionProgram
from llama_index.prompts.base import Prompt
from llama_index.types import Model


def _default_function_call(output_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Default OpenAI function to call."""
    schema = output_cls.schema()
    return {
        "name": schema["title"],
    }


class OpenAIPydanticProgram(BaseLLMFunctionProgram[OpenAI]):
    """
    An OpenAI-based function that returns a pydantic model.

    Note: this interface is not yet stable.
    """

    def __init__(
        self,
        output_cls: Type[Model],
        llm: OpenAI,
        prompt: Prompt,
        function_call: Union[str, Dict[str, Any]],
        verbose: bool = False,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose
        self._function_call = function_call

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[Model],
        prompt_template_str: str,
        llm: Optional[OpenAI] = None,
        verbose: bool = False,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "OpenAIPydanticProgram":
        llm = llm or OpenAI(model="gpt-3.5-turbo-0613")
        if not isinstance(llm, OpenAI):
            raise ValueError("llm must be a OpenAI instance")

        if not is_function_calling_model(llm.model):
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )

        prompt = Prompt(prompt_template_str)
        function_call = function_call or _default_function_call(output_cls)
        return cls(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            function_call=function_call,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        formatted_prompt = self._prompt.format(**kwargs)

        openai_fn_spec = to_openai_function(self._output_cls)

        chat_response = self._llm.chat(
            messages=[ChatMessage(role=MessageRole.USER, content=formatted_prompt)],
            functions=[openai_fn_spec],
            function_call=self._function_call,
        )
        message = chat_response.message
        if "function_call" not in message.additional_kwargs:
            raise ValueError(
                "Expected function call in ai_message.additional_kwargs, "
                "but none found."
            )

        function_call = message.additional_kwargs["function_call"]
        if self._verbose:
            name = function_call["name"]
            arguments_str = function_call["arguments"]
            print(f"Function call: {name} with args: {arguments_str}")

        output = self.output_cls.parse_raw(function_call["arguments"])
        return output

    async def acall(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> BaseModel:
        formatted_prompt = self._prompt.format(**kwargs)

        openai_fn_spec = to_openai_function(self._output_cls)

        chat_response = await self._llm.achat(
            messages=[ChatMessage(role=MessageRole.USER, content=formatted_prompt)],
            functions=[openai_fn_spec],
            function_call=self._function_call,
        )
        message = chat_response.message
        if "function_call" not in message.additional_kwargs:
            raise ValueError(
                "Expected function call in ai_message.additional_kwargs, "
                "but none found."
            )

        function_call = message.additional_kwargs["function_call"]
        if self._verbose:
            name = function_call["name"]
            arguments_str = function_call["arguments"]
            print(f"Function call: {name} with args: {arguments_str}")

        output = self.output_cls.parse_raw(function_call["arguments"])
        return output
