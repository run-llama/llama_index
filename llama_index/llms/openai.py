from typing import Any, Dict, Generator, Sequence

from pydantic import BaseModel, Field

from llama_index.llms.base import (
    LLM,
    ChatDeltaResponse,
    ChatResponse,
    ChatResponseType,
    CompletionDeltaResponse,
    CompletionResponse,
    CompletionResponseType,
    Message,
)
from llama_index.llms.generic_utils import messages_to_prompt, prompt_to_messages
from llama_index.llms.openai_utils import (
    completion_with_retry,
    is_chat_model,
    to_openai_message_dicts,
)


class OpenAI(LLM, BaseModel):
    model: str = Field("gpt-3.5-turbo")
    stream: bool = False
    max_retries: int = 10
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_chat_model(self) -> bool:
        return is_chat_model(self.model)

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseType:
        if self.is_chat_model:
            return self._chat(messages, **kwargs)
        else:
            prompt = messages_to_prompt(messages)
            return self._complete(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponseType:
        if self.is_chat_model:
            messages = prompt_to_messages(prompt)
            return self._chat(messages, **kwargs)
        else:
            return self._complete(prompt, **kwargs)

    def _get_all_kwargs(self, **kwargs) -> Dict[str, Any]:
        return {
            **self.additional_kwargs,
            **kwargs,
        }

    def _chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseType:
        if not self.is_chat_model:
            raise ValueError("This model is not a chat model.")

        message_dicts = to_openai_message_dicts(messages)
        params = self._get_all_kwargs(**kwargs)

        if not self.stream:
            response = completion_with_retry(
                model=self.model,
                max_retries=self.max_retries,
                messages=message_dicts,
                stream=self.stream,
                **params,
            )
            role = response["choices"][0]["message"]["role"]
            text = response["choices"][0]["message"]["content"]
            return ChatResponse(
                role=role,
                text=text,
            )
        else:

            def gen():
                text = ""
                for delta in completion_with_retry(
                    model=self.model,
                    max_retries=self.max_retries,
                    messages=message_dicts,
                    stream=self.stream,
                    **params,
                ):
                    role = delta["choices"][0]["delta"].get("role", "assistant")
                    delta = delta["choices"][0]["delta"].get("content", "")
                    text += delta
                    yield ChatDeltaResponse(
                        role=role,
                        delta=delta,
                        text=text,
                    )

            return gen()

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponseType:
        if self.is_chat_model:
            raise ValueError("This model is a chat model.")

        if not self.stream:
            response = completion_with_retry(
                model=self.model,
                prompt=prompt,
                max_retries=self.max_retries,
                stream=self.stream,
            )
            text = response["choices"][0]["text"]
            return CompletionResponse(
                text=text,
            )
        else:

            def gen():
                text = ""
                for response in completion_with_retry(
                    model=self.model,
                    prompt=prompt,
                    max_retries=self.max_retries,
                    stream=self.stream,
                ):
                    delta = response["choices"][0]["delta"].get("content", "")
                    text += delta
                    yield CompletionDeltaResponse(
                        delta=delta,
                        text=text,
                    )

            return gen()
