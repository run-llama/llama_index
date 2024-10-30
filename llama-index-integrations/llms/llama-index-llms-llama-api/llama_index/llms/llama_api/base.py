from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import chat_to_completion_decorator
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai.utils import (
    from_openai_message_dict,
    to_openai_message_dicts,
)
from llamaapi import LlamaAPI as Client


class LlamaAPI(CustomLLM):
    """LlamaAPI LLM.

    Examples:
        `pip install llama-index-llms-llama-api`

        ```python
        from llama_index.llms.llama_api import LlamaAPI

        # Obtain an API key from https://www.llama-api.com/
        api_key = "your-api-key"

        llm = LlamaAPI(api_key=api_key)

        # Call the complete method with a prompt
        resp = llm.complete("Paul Graham is ")

        print(resp)
        ```
    """

    model: str = Field(description="The llama-api model to use.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the llama-api API."
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = "llama-13b-chat",
        temperature: float = 0.1,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = Client(api_key)

    @classmethod
    def class_name(cls) -> str:
        return "llama_api_llm"

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_length": self.max_tokens,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=DEFAULT_NUM_OUTPUTS,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name="llama-api",
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts = to_openai_message_dicts(messages)
        json_dict = {
            "messages": message_dicts,
            **self._model_kwargs,
            **kwargs,
        }
        response = self._client.run(json_dict).json()
        message_dict = response["choices"][0]["message"]
        message = from_openai_message_dict(message_dict)

        return ChatResponse(message=message, raw=response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError("stream_complete is not supported for LlamaAPI")

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("stream_chat is not supported for LlamaAPI")
