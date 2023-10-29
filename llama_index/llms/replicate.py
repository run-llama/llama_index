from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)


class Replicate(CustomLLM):
    model: str = Field(description="The Replicate model to use.")
    image: str = Field(
        description="The image file for multimodal model to use. (optional)"
    )
    temperature: float = Field(description="The temperature to use for sampling.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    prompt_key: str = Field(description="The key to use for the prompt in API calls.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Replicate API."
    )
    is_chat_model: bool = Field(
        default=False, description="Whether the model is a chat model."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()

    def __init__(
        self,
        model: str,
        image: Optional[str] = "",
        temperature: float = 0.75,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        prompt_key: str = "prompt",
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        is_chat_model: bool = False,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)

        super().__init__(
            model=model,
            temperature=temperature,
            image=image,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            prompt_key=prompt_key,
            callback_manager=callback_manager,
            is_chat_model=is_chat_model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Replicate_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=self.is_chat_model,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_length": self.context_window,
        }
        if self.image != "":
            try:
                base_kwargs["image"] = open(self.image, "rb")
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Could not load image file. Please check whether the file exists"
                )
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {self.prompt_key: prompt, **self._model_kwargs, **kwargs}

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.complete(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self._messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response_gen = self.stream_complete(prompt, **kwargs)
        response_list = list(response_gen)
        final_response = response_list[-1]
        final_response.delta = None
        return final_response

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "Could not import replicate library."
                "Please install replicate with `pip install replicate`"
            )

        prompt = self._completion_to_prompt(prompt)
        input_dict = self._get_input_dict(prompt, **kwargs)
        response_iter = replicate.run(self.model, input=input_dict)

        def gen() -> CompletionResponseGen:
            text = ""
            for delta in response_iter:
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen()
