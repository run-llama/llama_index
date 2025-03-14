import json

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms.llm import ToolSelection
from llama_index.llms.openai_like import OpenAILike
from typing import Any, Dict, Optional, List
from llama_index.core.base.llms.generic_utils import (
     get_from_param_or_env,
)

DEFAULT_API_BASE = "https://api.novita.ai/v3/openai"
DEFAULT_MODEL = "deepseek/deepseek_v3"

def is_function_calling_model(model: str) -> bool:
    function_calling_models = {"deepseek_v3", "deepseek-r1-turbo", "deepseek-v3-turbo", "qwq-32b"}
    return any(model_name in model for model_name in function_calling_models)

class NovitaAI(OpenAILike):
    """NovitaAI LLM.
    Visit https://novita.ai to get more information about Novita.
    """

    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            api_key: Optional[str] = None,
            temperature: float = 0.95,
            max_tokens: int = 1024,
            is_chat_model: bool = True,
            additional_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        api_base = get_from_param_or_env("api_base", DEFAULT_API_BASE, "NOVITA_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "NOVITA_API_KEY")

        super().__init__(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model(model),
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NovitaAI"

    def get_tool_calls_from_response(
            self,
            response: "CompletionResponse",
            error_on_no_tool_call: bool = True,
            **kwargs: Any,
    ) -> List[ToolSelection]:
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            return []

        tool_selections = []
        for tool_call in tool_calls:
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=json.loads(tool_call.function.arguments),
                )
            )
        return tool_selections






