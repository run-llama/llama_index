import logging
from typing import (
    Any,
    Dict,
    Optional,
    Union,
    List,
)

from llama_index.core.types import Model

from llama_index.core.program.function_program import FunctionCallingProgram

_logger = logging.getLogger(__name__)


class JsonSchemaProgram(FunctionCallingProgram):
    """
    Json Schema Program.

    Uses Json Schema to generate a Pydantic model for the output.
    """

    def __call__(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Model, List[Model]]:
        llm_kwargs = self._prepare_schema(llm_kwargs)
        # print("Using JsonSchemaProgram")
        response = self._llm.chat(self._prepare_llm_messages(**kwargs), **llm_kwargs)
        return self._output_cls.model_validate_json(str(response.message.content))

    async def acall(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Model, List[Model]]:
        llm_kwargs = self._prepare_schema(llm_kwargs)
        # print("Using JsonSchemaProgram Async")
        response = await self._llm.achat(
            self._prepare_llm_messages(**kwargs), **llm_kwargs
        )
        return self._output_cls.model_validate_json(str(response.message.content))

    def _prepare_schema(self, llm_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            from openai.resources.beta.chat.completions import (
                _type_to_response_format as to_response_format,
            )
        except ImportError:
            raise ImportError("json_schema is not supported")
        llm_kwargs = llm_kwargs or {}
        llm_kwargs["response_format"] = to_response_format(self._output_cls)
        if "tool_choice" in llm_kwargs:
            del llm_kwargs["tool_choice"]
        return llm_kwargs

    def _prepare_llm_messages(self, **kwargs: Any):
        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
        return self._llm._extend_messages(messages)
