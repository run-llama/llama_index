from enum import Enum
from typing import Any, Sequence, Optional, Dict, Union, AsyncGenerator, Generator

from gigachat import GigaChat
from gigachat.models import Chat, Messages
from llama_index.core.base.llms.types import (
    LLMMetadata,
    CompletionResponse,
    CompletionResponseGen,
    ChatMessage,
    ChatResponse,
)
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import (
    llm_completion_callback,
    llm_chat_callback,
)
from llama_index.core.bridge.pydantic import Field


class GigaChatModel(str, Enum):
    GIGACHAT = "GigaChat"
    GIGACHAT_PLUS = "GigaChat-Plus"
    GIGACHAT_PRO = "GigaChat-Pro"


CONTEXT_WINDOWS = {
    GigaChatModel.GIGACHAT: 8192,
    GigaChatModel.GIGACHAT_PLUS: 32768,
    GigaChatModel.GIGACHAT_PRO: 8192,
}


class GigaChatLLM(CustomLLM):
    """
    GigaChat LLM Implementation.

    Examples:
        `pip install llama-index-llms-gigachat-ru`

        ```python
        from llama_index.llms.gigachat import GigaChatLLM

        llm = GigaChatLLM(
            credentials="YOUR_GIGACHAT_SECRET",
            verify_ssl_certs=False,
        )
        resp = llm.complete("What is the capital of France?")
        print(resp)
        ```
    """

    model: GigaChatModel = Field(default=GigaChatModel.GIGACHAT)
    base_url: Optional[str] = None
    auth_url: Optional[str] = None
    credentials: Optional[str] = None
    scope: Optional[str] = None
    access_token: Optional[str] = None
    profanity_check: Optional[bool] = None
    user: Optional[str] = None
    password: Optional[str] = None
    timeout: Optional[float] = None
    verify_ssl_certs: Optional[bool] = None
    verbose: Optional[bool] = None
    ca_bundle_file: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    key_file_password: Optional[str] = None

    @property
    def context_window(self) -> int:
        """Get context window."""
        return CONTEXT_WINDOWS[self.model]

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.context_window,
            model_name=self.model,
        )

    @llm_completion_callback()
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Get completion asynchronously."""
        async with GigaChat(**self._gigachat_kwargs) as giga:
            response = await giga.achat(
                Chat(
                    model=self.model,
                    messages=[Messages(role="user", content=prompt)],
                )
            )
        return CompletionResponse(
            text=response.choices[0].message.content,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Get completion."""
        with GigaChat(**self._gigachat_kwargs) as giga:
            response = giga.chat(
                Chat(
                    model=self.model,
                    messages=[Messages(role="user", content=prompt)],
                )
            )
        return CompletionResponse(
            text=response.choices[0].message.content,
        )

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Get chat asynchronously."""
        async with GigaChat(**self._gigachat_kwargs) as giga:
            response = await giga.achat(
                Chat(
                    model=self.model,
                    messages=[
                        Messages(role=message.role, content=message.content)
                        for message in messages
                    ],
                )
            )
        return ChatResponse(
            message=ChatMessage(
                content=response.choices[0].message.content,
                role="assistant",
            ),
        )

    @llm_chat_callback()
    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Get chat."""
        with GigaChat(**self._gigachat_kwargs) as giga:
            response = giga.chat(
                Chat(
                    model=self.model,
                    messages=[
                        Messages(role=message.role, content=message.content)
                        for message in messages
                    ],
                )
            )
        return ChatResponse(
            message=ChatMessage(
                content=response.choices[0].message.content,
                role="assistant",
            ),
        )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, Any]:
        """Get streaming completion asynchronously."""

        async def gen() -> AsyncGenerator[CompletionResponse, Any]:
            async with GigaChat(**self._gigachat_kwargs) as giga:
                chat = Chat(
                    model=self.model,
                    messages=[Messages(role="user", content=prompt)],
                )

                response = ""
                async for token in giga.astream(chat):
                    delta = token.choices[0].delta.content
                    response += delta
                    yield CompletionResponse(text=response, delta=delta)

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Get streaming completion."""

        def gen() -> Generator[CompletionResponse, Any, Any]:
            with GigaChat(**self._gigachat_kwargs) as giga:
                chat = Chat(
                    model=self.model,
                    messages=[Messages(role="user", content=prompt)],
                )

                response = ""
                for token in giga.stream(chat):
                    delta = token.choices[0].delta.content
                    response += delta
                    yield CompletionResponse(text=response, delta=delta)

        return gen()

    @property
    def _gigachat_kwargs(self) -> Dict[str, Union[str, bool, float]]:
        """Get GigaChat specific kwargs."""
        return {
            "base_url": self.base_url,
            "auth_url": self.auth_url,
            "credentials": self.credentials,
            "scope": self.scope,
            "access_token": self.access_token,
            "timeout": self.timeout,
            "verify_ssl_certs": self.verify_ssl_certs,
            "verbose": self.verbose,
            "ca_bundle_file": self.ca_bundle_file,
            "cert_file": self.cert_file,
            "key_file": self.key_file,
            "key_file_password": self.key_file_password,
        }

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "GigaChatLLM"
