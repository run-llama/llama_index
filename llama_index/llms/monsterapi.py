from typing import Any, Callable, Dict, Optional, Sequence
from pydantic import Field, PrivateAttr


from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import \
    messages_to_prompt as generic_messages_to_prompt


class MonsterLLM(CustomLLM):

    model: str = Field(description="The Predibase model to use.")
    monster_api_key: str = Field(description="The Predibase API key to use.")
    max_new_tokens: int = Field(
        description="The number of tokens to generate.")
    temperature: float = Field(
        description="The temperature to use for sampling.")
    context_window: int = Field(
        description="The number of context tokens available to the LLM."
    )
    messages_to_prompt: Optional[Callable] = None

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        monster_api_key: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable] = None,
    ) -> None:

        self._client, available_llms = self.initialize_client(monster_api_key)
        _messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        # Check if provided model is supported
        if model not in available_llms:
            raise RuntimeError(
                f"Model: {model} is not supported.Supported models are {available_llms}. Please update monsterapiclient to see if any models are added. pip install --upgrade monsterapi")

        super().__init__(
            model=model,
            monster_api_key=monster_api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            messages_to_prompt=_messages_to_prompt,
        )

    def initialize_client(self, monster_api_key: str) -> Any:
        try:
            from monsterapi.InputDataModels import MODEL_TYPES
            from monsterapi import client as MonsterClient
        except ImportError:
            raise ImportError(
                "Could not import Monster API client library."
                "Please install it with `pip install monsterapi`"
            )

        llm_models_enabled = [i for i, j in MODEL_TYPES.items() if j == "LLM"]

        return MonsterClient(monster_api_key), llm_models_enabled

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self.max_new_tokens,
            model_name=self._model,
        )

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_length": self.max_new_tokens,
            **self._additional_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Validate input args against input Pydantic model
        input_dict = self._get_input_dict(prompt, **kwargs)

        # Send request and receive process_id
        response = self._client.get_response(
            model=self._model, data=input_dict)
        process_id = response['process_id']

        # Wait for response and return result
        result = self._client.wait_and_get_result(process_id)

        return CompletionResponse(text=result['text'])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()
