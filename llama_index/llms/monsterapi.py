from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)

DEFAULT_MONSTER_TEMP = 0.75


class MonsterLLM(CustomLLM):
    model: str = Field(description="The MonsterAPI model to use.")
    monster_api_key: Optional[str] = Field(description="The MonsterAPI key to use.")
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_MONSTER_TEMP,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of context tokens available to the LLM.",
        gt=0,
    )

    messages_to_prompt: Callable = Field(
        description="The function to convert messages to a prompt.", exclude=True
    )
    completion_to_prompt: Callable = Field(
        description="The function to convert a completion to a prompt.", exclude=True
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model: str,
        monster_api_key: Optional[str] = None,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = DEFAULT_MONSTER_TEMP,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
    ) -> None:
        self._client, available_llms = self.initialize_client(monster_api_key)

        _messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        _completion_to_prompt = completion_to_prompt or (lambda x: x)

        # Check if provided model is supported
        if model not in available_llms:
            error_message = (
                f"Model: {model} is not supported. "
                f"Supported models are {available_llms}. "
                "Please update monsterapiclient to see if any models are added. "
                "pip install --upgrade monsterapi"
            )
            raise RuntimeError(error_message)

        super().__init__(
            model=model,
            monster_api_key=monster_api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            messages_to_prompt=_messages_to_prompt,
            completion_to_prompt=_completion_to_prompt,
        )

    def initialize_client(self, monster_api_key: Optional[str]) -> Any:
        try:
            from monsterapi import client as MonsterClient
            from monsterapi.InputDataModels import MODEL_TYPES
        except ImportError:
            raise ImportError(
                "Could not import Monster API client library."
                "Please install it with `pip install monsterapi`"
            )

        llm_models_enabled = [i for i, j in MODEL_TYPES.items() if j == "LLM"]

        return MonsterClient(monster_api_key), llm_models_enabled

    @classmethod
    def class_name(cls) -> str:
        return "MonsterLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model,
        )

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_length": self.max_new_tokens,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        return self.complete(prompt, formatted=True, **kwargs)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        # Validate input args against input Pydantic model
        input_dict = self._get_input_dict(prompt, **kwargs)

        # Send request and receive process_id
        response = self._client.get_response(model=self.model, data=input_dict)
        process_id = response["process_id"]

        # Wait for response and return result
        result = self._client.wait_and_get_result(process_id)

        return CompletionResponse(text=result["text"])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError
