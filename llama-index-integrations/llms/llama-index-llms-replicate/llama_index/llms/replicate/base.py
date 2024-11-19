from typing import Any, Dict, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)

DEFAULT_REPLICATE_TEMP = 0.75


class Replicate(CustomLLM):
    """Replicate LLM.

    Examples:
        `pip install llama-index-llms-replicate`

        ```python
        from llama_index.llms.replicate import Replicate

        # Set up the Replicate API token
        import os
        os.environ["REPLICATE_API_TOKEN"] = "<your API key>"

        # Initialize the Replicate class
        llm = Replicate(
            model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
        )

        # Example of calling the 'complete' method with a prompt
        resp = llm.complete("Who is Paul Graham?")

        print(resp)
        ```
    """

    model: str = Field(description="The Replicate model to use.")
    temperature: float = Field(
        default=DEFAULT_REPLICATE_TEMP,
        description="The temperature to use for sampling.",
        ge=0.01,
        le=1.0,
    )
    image: str = Field(
        default="", description="The image file for multimodal model to use. (optional)"
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    prompt_key: str = Field(
        default="prompt", description="The key to use for the prompt in API calls."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Replicate API."
    )
    is_chat_model: bool = Field(
        default=False, description="Whether the model is a chat model."
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
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response_gen = self.stream_complete(prompt, formatted=formatted, **kwargs)
        response_list = list(response_gen)
        final_response = response_list[-1]
        final_response.delta = None
        return final_response

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        try:
            import replicate
        except ImportError:
            raise ImportError(
                "Could not import replicate library."
                "Please install replicate with `pip install replicate`"
            )

        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        input_dict = self._get_input_dict(prompt, **kwargs)
        response_iter = replicate.stream(self.model, input=input_dict)

        def gen() -> CompletionResponseGen:
            text = ""
            for server_event in response_iter:
                delta = str(server_event)
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                )

        return gen()
