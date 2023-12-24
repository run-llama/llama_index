import json
from typing import Any, Dict, Sequence

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.custom import CustomLLM
from llama_index.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)


class Ollama(CustomLLM):
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
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
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,  # Ollama supports chat API for all models
            is_function_calling_model=True,  # Ollama supports json format outputs
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )

        with requests.post(
            url=f"{self.base_url}/api/chat/",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                        **message.additional_kwargs,
                    }
                    for message in messages
                ],
                "options": self._model_kwargs,
                "stream": False,
                **kwargs,
            },
        ) as response:
            response.raise_for_status()
            message = response.json()["message"]
            return ChatResponse(
                message=ChatMessage(
                    content=message.get("content", ""),
                    role=message.get("role", MessageRole.ASSISTANT.value),
                    additional_kwargs={
                        k: v
                        for k, v in message.items()
                        if k != "content" and k != "role"
                    },
                ),
                raw=response.json(),
                additional_kwargs={
                    k: v for k, v in response.json().items() if k != "message"
                },
            )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )

        with requests.post(
            url=f"{self.base_url}/api/chat/",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                        **message.additional_kwargs,
                    }
                    for message in messages
                ],
                "options": self._model_kwargs,
                "stream": True,
                **kwargs,
            },
            stream=True,
        ) as response:
            response.raise_for_status()
            text = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)
                    delta = chunk["message"].get("content", "")
                    role = chunk["message"].get("role", MessageRole.ASSISTANT.value)
                    text += delta
                    yield ChatResponse(
                        message=ChatMessage(
                            message=text,
                            role=role,
                            additional_kwargs={
                                k: v
                                for k, v in chunk["message"].items()
                                if k != "content" and k != "role"
                            },
                        ),
                        delta=delta,
                        raw=chunk,
                        additional_kwargs={
                            k: v for k, v in chunk.items() if k != "message"
                        },
                    )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )

        with requests.post(
            url=f"{self.base_url}/api/generate/",
            json={
                self.prompt_key: prompt,
                "model": self.model,
                "options": self._model_kwargs,
                "stream": False,
                **kwargs,
            },
        ) as response:
            response.raise_for_status()
            return CompletionResponse(
                text=response.json().get("response", ""),
                raw=response.json(),
            )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )

        with requests.post(
            url=f"{self.base_url}/api/generate/",
            json={
                self.prompt_key: prompt,
                "model": self.model,
                "options": self._model_kwargs,
                "stream": True,
                **kwargs,
            },
            stream=True,
        ) as response:
            response.raise_for_status()
            text = ""

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)
                    delta = chunk.get("response", "")
                    text += delta
                    yield CompletionResponse(
                        delta=delta,
                        text=text,
                        raw=chunk,
                    )
