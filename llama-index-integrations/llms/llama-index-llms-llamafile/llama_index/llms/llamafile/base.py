import io
import json
from typing import Any, Dict, Sequence, Tuple

import httpx
from httpx import Timeout

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

DEFAULT_REQUEST_TIMEOUT = 30.0


def get_addtional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


class Llamafile(CustomLLM):
    base_url: str = Field(
        default="http://localhost:8080",
        description="Base url where the llamafile server is listening.",
    )

    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )

    #
    # Generation options
    #
    temperature: float = Field(
        default=0.8,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )

    seed: int = Field(
        default=0,
        description="Random seed"
    )

    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options to pass in requests to the llamafile API.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "llamafile_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            is_chat_model=True,  # llamafile has OpenAI-compatible chat API for all models
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "seed": self.seed,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = {
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    **message.additional_kwargs,
                }
                for message in messages
            ],
            "options": self._model_kwargs,
            "stream": False,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            message = choice["message"]

            return ChatResponse(
                message=ChatMessage(
                    content=message.get("content"),
                    role=MessageRole(message.get("role")),
                    additional_kwargs=get_addtional_kwargs(
                        message, ("content", "role")
                    ),
                ),
                raw=raw,
                additional_kwargs=get_addtional_kwargs(raw, ("choice",)),
            )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = {
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    **message.additional_kwargs,
                }
                for message in messages
            ],
            "options": self._model_kwargs,
            "stream": True,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:

                response.raise_for_status()

                with io.StringIO() as buff:
                    for line in response.iter_lines():
                        if line:
                            chunk = self._get_streaming_chunk_content(line)
                            choice = chunk.pop("choices")[0]
                            delta_message = choice["delta"]

                            # default to 'assistant' if response does not contain 'role'
                            role = delta_message.get("role", MessageRole.ASSISTANT)

                            # The last message has no content
                            delta_content = delta_message.get("content", None)
                            if delta_content:
                                buff.write(delta_content)
                            else:
                                delta_content = ""

                            yield ChatResponse(
                                message=ChatMessage(
                                    content=buff.getvalue(),
                                    role=MessageRole(role),
                                    additional_kwargs=get_addtional_kwargs(
                                        delta_message, ("content", "role")
                                    ),
                                ),
                                delta=delta_content,
                                raw=chunk,
                                additional_kwargs=get_addtional_kwargs(
                                    chunk, ("choices",)
                                ),
                            )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        payload = {
            "prompt": prompt,
            "stream": False,
            **self._model_kwargs,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/completion",
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
            text = raw.get("content")
            return CompletionResponse(
                text=text,
                raw=raw,
                additional_kwargs=get_addtional_kwargs(raw, ("response",)),
            )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        payload = {
            "prompt": prompt,
            "stream": True,
            **self._model_kwargs,
            **kwargs,
        }

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            with client.stream(
                method="POST",
                url=f"{self.base_url}/completion",
                json=payload,
            ) as response:
                response.raise_for_status()

                with io.StringIO() as buff:
                    for line in response.iter_lines():
                        if line:
                            chunk = self._get_streaming_chunk_content(line)
                            delta = chunk.get("content")
                            buff.write(delta)
                            yield CompletionResponse(
                                delta=delta,
                                text=buff.getvalue(),
                                raw=chunk,
                                additional_kwargs=get_addtional_kwargs(
                                    chunk, ("content",)
                                ),
                            )

    def _get_streaming_chunk_content(self, chunk: str) -> Dict:
        """When streaming is turned on, llamafile server returns lines like:

        'data: {"content":" They","multimodal":true,"slot_id":0,"stop":false}'

        Here, we convert this to a dict and return the value of the 'content'
        field
        """

        if chunk.startswith("data:"):
            cleaned = chunk.lstrip("data: ")
            data = json.loads(cleaned)
            return data
        else:
            raise ValueError(
                f"Received chunk with unexpected format during streaming: '{chunk}'"
            )
