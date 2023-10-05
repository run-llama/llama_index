import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from llama_index.bridge.pydantic import Field
from llama_index.callbacks import CallbackManager
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    llm_chat_callback,
    llm_completion_callback,
)


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
        default_factory=dict, description="Additional kwargs for the Replicate API."
    )
    base_url: str = Field(
        description="The address of your target model served by rungpt."
    )

    def __init__(
        self,
        model: Optional[str] = "rungpt",
        endpoint: str = "0.0.0.0:51002",
        temperature: float = 0.75,
        max_tokens: Optional[int] = 256,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        if endpoint.startswith("http://"):
            base_url = endpoint
        else:
            base_url = "http://" + endpoint
        super().__init__(
            model=model,
            endpoint=endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
            base_url=base_url,
        )

    @classmethod
    def class_name(cls) -> str:
        return "RunGptLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self._model,
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
        response_gpt = requests.post(
            self.base_url + "/generate",
            json=self._request_pack("complete", prompt, **kwargs),
            stream=False,
        ).json()

        return CompletionResponse(
            text=response_gpt["choices"][0]["text"],
            additional_kwargs=response_gpt["usage"],
            raw=response_gpt,
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
        response_gpt = requests.post(
            self.base_url + "/generate_stream",
            json=self._request_pack("complete", prompt, **kwargs),
            stream=True,
        )
        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "Could not import sseclient-py library."
                "Please install requests with `pip install sseclient-py`"
            )
        client = sseclient.SSEClient(response_gpt)
        response_iter = client.events()

        def gen() -> CompletionResponseGen:
            text = ""
            for item in response_iter:
                item_dict = json.loads(json.dumps(eval(item.data)))
                delta = item_dict["choices"][0]["text"]
                additional_kwargs = item_dict["usage"]
                text = text + self._space_handler(delta)
                yield CompletionResponse(
                    text=text,
                    delta=delta,
                    raw=item_dict,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_list = self._message_wrapper(messages)
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )
        response_gpt = requests.post(
            self.base_url + "/chat",
            json=self._request_pack("chat", message_list, **kwargs),
            stream=False,
        ).json()
        chat_message, _ = self._message_unpacker(response_gpt)
        return ChatResponse(message=chat_message, raw=response_gpt)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        message_list = self._message_wrapper(messages)
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Could not import requests library."
                "Please install requests with `pip install requests`"
            )
        response_gpt = requests.post(
            self.base_url + "/chat_stream",
            json=self._request_pack("chat", message_list, **kwargs),
            stream=True,
        )
        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "Could not import sseclient-py library."
                "Please install requests with `pip install sseclient-py`"
            )
        client = sseclient.SSEClient(response_gpt)
        chat_iter = client.events()

        def gen() -> ChatResponseGen:
            content = ""
            for item in chat_iter:
                item_dict = json.loads(json.dumps(eval(item.data)))
                chat_message, delta = self._message_unpacker(item_dict)
                content = content + self._space_handler(delta)
                chat_message.content = content
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

        return gen()

    def _message_wrapper(self, messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
        message_list = []
        for message in messages:
            role = message.role.value
            content = message.content
            message_list.append({"role": role, "content": content})
        return message_list

    def _message_unpacker(
        self, response_gpt: Dict[str, Any]
    ) -> Tuple[ChatMessage, str]:
        message = response_gpt["choices"][0]["message"]
        additional_kwargs = response_gpt["usage"]
        role = message["role"]
        content = message["content"]
        key = MessageRole.SYSTEM
        for r in MessageRole:
            if r.value == role:
                key = r
        chat_message = ChatMessage(
            role=key, content=content, additional_kwargs=additional_kwargs
        )
        return chat_message, content

    def _request_pack(
        self, mode: str, prompt: Union[str, List[Dict[str, Any]]], **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
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
        return None

    def _space_handler(self, word: str) -> str:
        if word.isalnum():
            return " " + word
        return word
