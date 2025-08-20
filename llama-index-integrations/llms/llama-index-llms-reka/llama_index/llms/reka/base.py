import os
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

try:
    from reka.client import Reka, AsyncReka
    from reka.core import ApiError
except ImportError:
    raise ValueError(
        "Reka is not installed. Please install it with `pip install reka-api`."
    )


DEFAULT_REKA_MODEL = "reka-flash"
DEFAULT_REKA_MAX_TOKENS = 512
DEFAULT_REKA_CONTEXT_WINDOW = 128000


def process_messages_for_reka(messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
    reka_messages = []
    system_message = None

    for message in messages:
        if message.role == MessageRole.SYSTEM:
            if system_message is None:
                system_message = message.content
            else:
                raise ValueError("Multiple system messages are not supported.")
        elif message.role == MessageRole.USER:
            content = message.content
            if system_message:
                content = f"{system_message}\n{content}"
                system_message = None
            reka_messages.append({"role": "user", "content": content})
        elif message.role == MessageRole.ASSISTANT:
            reka_messages.append({"role": "assistant", "content": message.content})
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    return reka_messages


class RekaLLM(CustomLLM):
    """Reka LLM integration for LlamaIndex."""

    model: str = Field(default=DEFAULT_REKA_MODEL, description="The Reka model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_REKA_MAX_TOKENS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for Reka API calls.",
    )

    _client: Reka = PrivateAttr()
    _aclient: AsyncReka = PrivateAttr()

    def __init__(
        self,
        model: str = DEFAULT_REKA_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_REKA_MAX_TOKENS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        Initialize the RekaLLM instance.

        Args:
            model (str): The Reka model to use, choose from ['reka-flash', 'reka-core', 'reka-edge'].
            api_key (Optional[str]): The API key for Reka.
            temperature (float): The temperature to use for sampling.
            max_tokens (int): The maximum number of tokens to generate.
            additional_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for Reka API calls.
            callback_manager (Optional[CallbackManager]): A callback manager for handling callbacks.

        Raises:
            ValueError: If the Reka API key is not provided and not set in the environment.

        Example:
            >>> reka_llm = RekaLLM(
            ...     model="reka-flash",
            ...     api_key="your-api-key-here",
            ...     temperature=0.7,
            ...     max_tokens=100
            ... )

        """
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        api_key = api_key or os.getenv("REKA_API_KEY")
        if not api_key:
            raise ValueError(
                "Reka API key is required. Please provide it as an argument or set the REKA_API_KEY environment variable."
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
        )
        self._client = Reka(api_key=api_key)
        self._aclient = AsyncReka(api_key=api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=DEFAULT_REKA_CONTEXT_WINDOW,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._model_kwargs, **kwargs}

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Send a chat request to the Reka API.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ChatResponse: The response from the Reka API.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> reka_llm = RekaLLM(api_key="your-api-key-here")
            >>> messages = [
            ...     ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ...     ChatMessage(role=MessageRole.USER, content="What's the capital of France?")
            ... ]
            >>> response = reka_llm.chat(messages)
            >>> print(response.message.content)

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            response = self._client.chat.create(messages=reka_messages, **all_kwargs)
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.responses[0].message.content,
                ),
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Send a completion request to the Reka API.

        Args:
            prompt (str): The prompt for completion.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            CompletionResponse: The response from the Reka API.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> reka_llm = RekaLLM(api_key="your-api-key-here")
            >>> response = reka_llm.complete("The capital of France is")
            >>> print(response.text)

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            response = self._client.chat.create(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
            return CompletionResponse(
                text=response.responses[0].message.content,
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Send a streaming chat request to the Reka API.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ChatResponseGen: A generator yielding chat responses.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> reka_llm = RekaLLM(api_key="your-api-key-here")
            >>> messages = [
            ...     ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ...     ChatMessage(role=MessageRole.USER, content="Tell me a short story.")
            ... ]
            >>> for chunk in reka_llm.stream_chat(messages):
            ...     print(chunk.delta, end="", flush=True)

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            stream = self._client.chat.create_stream(
                messages=reka_messages, **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        def gen() -> ChatResponseGen:
            prev_content = ""
            for chunk in stream:
                content = chunk.responses[0].chunk.content
                content_delta = content[len(prev_content) :]
                prev_content = content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Send a streaming completion request to the Reka API.

        Args:
            prompt (str): The prompt for completion.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            CompletionResponseGen: A generator yielding completion responses.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> reka_llm = RekaLLM(api_key="your-api-key-here")
            >>> prompt = "Write a haiku about programming:"
            >>> for chunk in reka_llm.stream_complete(prompt):
            ...     print(chunk.delta, end="", flush=True)

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            stream = self._client.chat.create_stream(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        def gen() -> CompletionResponseGen:
            prev_text = ""
            for chunk in stream:
                text = chunk.responses[0].chunk.content
                text_delta = text[len(prev_text) :]
                prev_text = text
                yield CompletionResponse(
                    text=text,
                    delta=text_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """
        Send an asynchronous chat request to the Reka API.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ChatResponse: The response from the Reka API.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> import asyncio
            >>> from llama_index.llms.reka import RekaLLM
            >>> from llama_index.core.base.llms.types import ChatMessage, MessageRole
            >>>
            >>> async def main():
            ...     reka_llm = RekaLLM(api_key="your-api-key-here")
            ...     messages = [
            ...         ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ...         ChatMessage(role=MessageRole.USER, content="What's the meaning of life?")
            ...     ]
            ...     response = await reka_llm.achat(messages)
            ...     print(response.message.content)
            >>>
            >>> asyncio.run(main())

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            response = await self._aclient.chat.create(
                messages=reka_messages, **all_kwargs
            )
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.responses[0].message.content,
                ),
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Send an asynchronous completion request to the Reka API.

        Args:
            prompt (str): The prompt for completion.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            CompletionResponse: The response from the Reka API.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> import asyncio
            >>> from llama_index.llms.reka import RekaLLM
            >>>
            >>> async def main():
            ...     reka_llm = RekaLLM(api_key="your-api-key-here")
            ...     prompt = "The capital of France is"
            ...     response = await reka_llm.acomplete(prompt)
            ...     print(response.text)
            >>>
            >>> asyncio.run(main())

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            response = await self._aclient.chat.create(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
            return CompletionResponse(
                text=response.responses[0].message.content,
                raw=response.__dict__,
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """
        Send an asynchronous streaming chat request to the Reka API.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            ChatResponseAsyncGen: An asynchronous generator yielding chat responses.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> import asyncio
            >>> from llama_index.llms.reka import RekaLLM
            >>> from llama_index.core.base.llms.types import ChatMessage, MessageRole
            >>>
            >>> async def main():
            ...     reka_llm = RekaLLM(api_key="your-api-key-here")
            ...     messages = [
            ...         ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ...         ChatMessage(role=MessageRole.USER, content="Tell me a short story about a robot.")
            ...     ]
            ...     async for chunk in await reka_llm.astream_chat(messages):
            ...         print(chunk.delta, end="", flush=True)
            ...     print()  # New line after the story is complete
            >>>
            >>> asyncio.run(main())

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        reka_messages = process_messages_for_reka(messages)

        try:
            stream = self._aclient.chat.create_stream(
                messages=reka_messages, **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        async def gen() -> ChatResponseAsyncGen:
            prev_content = ""
            async for chunk in stream:
                content = chunk.responses[0].chunk.content
                content_delta = content[len(prev_content) :]
                prev_content = content
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                    ),
                    delta=content_delta,
                    raw=chunk.__dict__,
                )

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """
        Send an asynchronous streaming completion request to the Reka API.

        Args:
            prompt (str): The prompt for completion.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            CompletionResponseAsyncGen: An asynchronous generator yielding completion responses.

        Raises:
            ValueError: If there's an error with the Reka API call.

        Example:
            >>> import asyncio
            >>> from llama_index.llms.reka import RekaLLM
            >>>
            >>> async def main():
            ...     reka_llm = RekaLLM(api_key="your-api-key-here")
            ...     prompt = "Write a haiku about artificial intelligence:"
            ...     async for chunk in await reka_llm.astream_complete(prompt):
            ...         print(chunk.delta, end="", flush=True)
            ...     print()  # New line after the haiku is complete
            >>>
            >>> asyncio.run(main())

        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        try:
            stream = self._aclient.chat.create_stream(
                messages=[{"role": "user", "content": prompt}], **all_kwargs
            )
        except ApiError as e:
            raise ValueError(f"Reka API error: {e.status_code} - {e.body}")

        async def gen() -> CompletionResponseAsyncGen:
            prev_text = ""
            async for chunk in stream:
                text = chunk.responses[0].chunk.content
                text_delta = text[len(prev_text) :]
                prev_text = text
                yield CompletionResponse(
                    text=text,
                    delta=text_delta,
                    raw=chunk.__dict__,
                )

        return gen()
