from typing import Any, Callable, Dict, Optional, Sequence

import ai21
from ai21 import AI21Client, Tokenizer
from llama_index.core.base.llms.generic_utils import (
    completion_to_chat_decorator,
)
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
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index_client import MessageRole

from llama_index.llms.ai21.utils import ai21_model_to_context_size, message_to_ai21_message

_DEFAULT_AI21_MODEL = "jamba-instruct"
_TOKENIZER_NAME_FORMAT = "{model_name}-tokenizer"


class AI21(CustomLLM):
    """AI21 Labs LLM.

    Examples:
        `pip install llama-index-llms-ai21`

        ```python
        from llama_index.llms.ai21 import AI21

        llm = AI21(model="j2-mid", api_key=api_key)
        resp = llm.complete("Paul Graham is ")
        print(resp)
        ```
    """

    model: str = Field(
        description="The AI21 model to use.", default=_DEFAULT_AI21_MODEL
    )
    max_tokens: int = Field(description="The maximum number of tokens to generate.")
    temperature: float = Field(description="The temperature to use for sampling.")

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the anthropic API."
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str = _DEFAULT_AI21_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = 512,
        max_retries: int = 10,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        self._client = AI21Client(
            api_key=api_key,
            api_host=base_url,
            timeout_sec=timeout,
            num_retries=max_retries,
            headers=default_headers,
            via="llama-index",
        )

        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get Class Name."""
        return "AI21_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=ai21_model_to_context_size(self.model),
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return self._client.get_tokenizer(
            _TOKENIZER_NAME_FORMAT.format(model_name=self.model)
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        ai21.api_key = self._api_key

        response = ai21.Completion.execute(**all_kwargs, prompt=prompt)

        return CompletionResponse(
            text=response["completions"][0]["data"]["text"], raw=response.__dict__
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError(
            "AI21 does not currently support streaming completion."
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        messages = [message_to_ai21_message(message) for message in messages]
        response = self._client.chat.completions.create(
            messages=messages,
            stream=False,
            **all_kwargs,
        )

        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
            ),
            raw=dict(response),
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("AI21 does not Currently Support Streaming Chat.")
