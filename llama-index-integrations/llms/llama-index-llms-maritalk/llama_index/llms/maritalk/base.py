from typing import Any, Dict, Optional, Sequence
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import llm_chat_callback

import requests

class Maritalk(LLM):
    api_key: str = Field(description="Your MariTalk API key.")
    temperature: float = Field(
        default=0.7, gt=0.0, lt=1.0, description="Run inference with this temperature. Must be in the closed interval [0.0, 1.0]."
    )
    max_tokens: int = Field(
        default=512, gt=0, description="The maximum number of tokens to generate in the reply."
    )
    do_sample: bool = Field(
        default=True, description="Whether or not to use sampling; use `True` to enable."
    )
    top_p: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Nucleus sampling parameter controlling the size of the probability mass considered for sampling."
    )
    system_message_workaround: bool = Field(
        default=True, description="Whether to include a workaround for system messages by adding them as a user message."
    )

    _endpoint: str = PrivateAttr("https://chat.maritaca.ai/api/chat/inference")

    @classmethod
    def class_name(cls) -> str:
        return "Maritalk"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="maritalk",
            context_window=self.max_tokens,
            is_chat_model=True,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Prepare the data payload for the Maritalk API
        formatted_messages = [
            {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
            for msg in messages
        ]

        data = {
            "messages": formatted_messages,
            "do_sample": self.do_sample,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        headers = {"authorization": f"Key {self.api_key}"}

        response = requests.post(self._endpoint, json=data, headers=headers)
        if response.status_code == 429:
            return ChatResponse(
                message=ChatMessage(role=MessageRole.SYSTEM, content="Rate limited, please try again soon"),
                raw=response.text,
            )
        elif response.ok:
            answer = response.json()["answer"]
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                raw=response.json(),
            )
        else:
            response.raise_for_status()
