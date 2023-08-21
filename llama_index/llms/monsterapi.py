from typing import Any, Callable, Dict, Optional, Sequence

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
    def __init__(
        self,
        model: str,
        monster_api_key: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        messages_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.monster_api_key = monster_api_key
        self.max_new_tokens = max_new_tokens
        self._model = model
        self._context_window = context_window
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self.callback_manager = callback_manager or CallbackManager([])

        # model kwargs
        self._temperature = temperature
        self._additional_kwargs = additional_kwargs or {}

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
            "temperature": self._temperature,
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
        try:
            from monsterapi.InputDataModels import MODEL_TYPES
            from monsterapi import client as MonsterClient
        except ImportError:
            raise ImportError(
                "Could not import Monster API client library."
                "Please install it with `pip install monsterapi`"
            )

        # Check if model is supported
        llm_models_enabled = [i for i, j in MODEL_TYPES.items() if j == "LLM"]
        if self._model not in llm_models_enabled:
            raise RuntimeError(
                f"Model: {self._model} is not supported.Supported models are {llm_models_enabled}. Please update monsterapiclient to see if any models are added. pip install --upgrade monsterapi")

        # Validate input args against input Pydantic model
        input_dict = self._get_input_dict(prompt, **kwargs)

        # Initiate client object
        monster_client = MonsterClient(api_key=self.monster_api_key)

        # Send request and receive process_id
        response = monster_client.get_response(
            model=self._model, data=input_dict)
        process_id = response['process_id']

        # Wait for response and return result
        result = monster_client.wait_and_get_result(process_id)

        return CompletionResponse(text=result['text'])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()
