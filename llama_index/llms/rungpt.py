from typing import Any, Optional, Sequence, Dict
from pydantic import Field
import json
import requests
import sseclient
from llama_index.llms.base import (
    LLM,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    MessageRole,
    CompletionResponse,
    ChatResponseAsyncGen,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    llm_completion_callback,
    llm_chat_callback,
)
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.callbacks import CallbackManager


class RunGptLLM(LLM):
    """The opengpt of Jina AI models."""

    model: Optional[str] = Field(description="The rungpt model to use.")
    endpoint: str = Field(description="The endpoint of serving address.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_tokens: Optional[int] = Field(description="Max tokens model generates.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additonal kwargs for the Replicate API."
    )
    base_url: str = Field(
        description="The address of your target model served by rungpt."
    )

    def __init__(
        self,
        model: Optional[str] = "rungpt",
        endpoint: Optional[str] = "0.0.0.0:51002",
        temperature: float = 0.75,
        max_tokens: Optional[int] = 256,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            model=model,
            endpoint=endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
            base_url="http://" + endpoint,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self._model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:

        response_gpt = requests.post(
            self.base_url + "/generate",
            json=self._request_pack("complete", prompt, **kwargs),
            stream=False,
        ).json()

        response_gpt = CompletionResponse(
            text=response_gpt["choices"][0]["text"],
            additional_kwargs=response_gpt["usage"],
            raw=response_gpt,
        )
        return response_gpt

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:

        response_gpt = requests.post(
            self.base_url + "/generate_stream",
            json=self._request_pack("complete", prompt, **kwargs),
            stream=True,
        )

        client = sseclient.SSEClient(response_gpt)
        response_iter = client.events()

        def gen() -> CompletionResponseGen:
            text = ""
            for item in response_iter:
                item_dict = json.loads(json.dumps(eval(item.data)))
                delta = item_dict["choices"][0]["text"]
                text = text + delta
                yield CompletionResponse(text=text, delta=delta, raw=item_dict)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_list = list()
        for message in messages:
            role = message.role.value
            content = message.content
            message_list.append({"role": role, "content": content})
        response_gpt = requests.post(
            self.base_url + "/chat",
            json=self._request_pack("chat", message_list, **kwargs),
            stream=False,
        ).json()
        message = response_gpt["choices"][0]["message"]
        role = message["role"]
        content = message["content"]
        key = MessageRole.SYSTEM
        for r in MessageRole:
            if r.value == role:
                key = r
        chat_message = ChatMessage(role=key, content=content)
        response_gpt = ChatResponse(message=chat_message, raw=response_gpt)
        return response_gpt

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_list = list()
        for message in messages:
            role = message.role.value
            content = message.content
            message_list.append({"role": role, "content": content})
        response_gpt = requests.post(
            self.base_url + "/chat_stream",
            json=self._request_pack("chat", message_list, **kwargs),
            stream=True,
        )
        client = sseclient.SSEClient(response_gpt)
        chat_iter = client.events()

        def gen() -> ChatResponse:
            content = ""
            for item in chat_iter:
                item_dict = json.loads(json.dumps(eval(item.data)))
                message = item_dict["choices"][0]["message"]
                role = message["role"]
                delta = message["content"]
                content = content + delta
                key = MessageRole.SYSTEM
                for r in MessageRole:
                    if r.value == role:
                        key = r
                chat_message = ChatMessage(role=key, content=content)
                yield ChatResponse(message=chat_message, raw=item_dict, delta=delta)

        return gen()

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    def _request_pack(self, mode: str, prompt: str, **kwargs: Any) -> dict:
        if mode == "complete":
            return {
                "prompt": prompt,
                "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
                "temperature": kwargs.pop("temperature", self.temperature),
                "top_k": kwargs.pop("top_k", 50),
                "top_p": kwargs.pop("top_p", 0.95),
                "repetition_penalty": kwargs.pop("repetition_penalty", 1.2),
                "do_sample": kwargs.pop("do_sample", False),
                "echo": kwargs.pop("echo", True),
                "n": kwargs.pop("n", 1),
                "stop": kwargs.pop("stop", "."),
            }
        elif mode == "chat":
            return {
                "messages": prompt,
                "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
                "temperature": kwargs.pop("temperature", self.temperature),
                "top_k": kwargs.pop("top_k", 50),
                "top_p": kwargs.pop("top_p", 0.95),
                "repetition_penalty": kwargs.pop("repetition_penalty", 1.2),
                "do_sample": kwargs.pop("do_sample", False),
                "echo": kwargs.pop("echo", True),
                "n": kwargs.pop("n", 1),
                "stop": kwargs.pop("stop", "."),
            }
