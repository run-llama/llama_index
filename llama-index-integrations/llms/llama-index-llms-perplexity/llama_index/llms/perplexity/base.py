import json
from typing import Any, Callable, Dict, Optional, Sequence

import httpx
import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai.utils import to_openai_message_dicts


class Perplexity(LLM):
    """
    Perplexity LLM.

    Examples:
        `pip install llama-index-llms-perplexity`

        ```python
        from llama_index.llms.perplexity import Perplexity
        from llama_index.core.llms import ChatMessage

        pplx_api_key = "your-perplexity-api-key"

        llm = Perplexity(
            api_key=pplx_api_key, model="llama-3.1-sonar-small-128k-online", temperature=0.5
        )

        messages_dict = [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": "Tell me 5 sentences about Perplexity."},
        ]
        messages = [ChatMessage(**msg) for msg in messages_dict]

        response = llm.chat(messages)
        print(str(response))
        ```
    """

    model: str = Field(description="The Perplexity model to use.")
    temperature: float = Field(description="The temperature to use during generation.")
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    context_window: Optional[int] = Field(
        default=None,
        description="The context window to use during generation.",
    )
    api_key: str = Field(
        default=None, description="The Perplexity API key.", exclude=True
    )
    api_base: str = Field(
        default="https://api.perplexity.ai",
        description="The base URL for Perplexity API.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Perplexity API."
    )
    max_retries: int = Field(
        default=10, description="The maximum number of API retries."
    )
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Headers for API requests."
    )

    def __init__(
        self,
        model: str = "llama-3.1-sonar-small-128k-online",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.perplexity.ai",
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        context_window: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            api_key=api_key,
            api_base=api_base,
            headers=headers,
            context_window=context_window,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "perplexity_llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=(
                self.context_window
                if self.context_window is not None
                else self._get_context_window()
            ),
            num_output=self.max_tokens
            or -1,  # You can replace this with the appropriate value
            is_chat_model=self._is_chat_model(),
            model_name=self.model,
        )

    def _get_context_window(self) -> int:
        # Check https://docs.perplexity.ai/guides/model-cards for latest model information
        model_context_windows = {
            "llama-3.1-sonar-small-128k-online": 127072,
            "llama-3.1-sonar-large-128k-online": 127072,
            "llama-3.1-sonar-huge-128k-online": 127072,
        }
        return model_context_windows.get(
            self.model, 127072
        )  # Default to 127072 tokens if model is not found

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Get all data for the request as a dictionary."""
        base_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            base_kwargs["max_tokens"] = self.max_tokens
        return {**base_kwargs, **self.additional_kwargs, **kwargs}

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            **self._get_all_kwargs(**kwargs),
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return CompletionResponse(
            text=data["choices"][0]["message"]["content"], raw=data
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, **kwargs)

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)
        payload = {
            "model": self.model,
            "messages": message_dicts,
            **self._get_all_kwargs(**kwargs),
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        message = ChatMessage(
            role="assistant", content=data["choices"][0]["message"]["content"]
        )
        return ChatResponse(message=message, raw=data)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat(messages, **kwargs)

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            **self._get_all_kwargs(**kwargs),
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return CompletionResponse(
            text=data["choices"][0]["message"]["content"], raw=data
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await self._acomplete(prompt, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)
        payload = {
            "model": self.model,
            "messages": message_dicts,
            **self._get_all_kwargs(**kwargs),
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        message = ChatMessage(
            role="assistant", content=data["choices"][0]["message"]["content"]
        )
        return ChatResponse(message=message, raw=data)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return await self._achat(messages, **kwargs)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        payload = {
            "model": self.model,
            # "prompt": prompt,
            "messages": messages,
            "stream": True,
            **self._get_all_kwargs(**kwargs),
        }
        print(payload)

        def gen() -> CompletionResponseGen:
            with requests.Session() as session:
                with session.post(
                    url, json=payload, headers=self.headers, stream=True
                ) as response:
                    response.raise_for_status()
                    text = ""
                    for line in response.iter_lines(
                        decode_unicode=True
                    ):  # decode lines to Unicode
                        if line.startswith("data:"):
                            data = json.loads(line[5:])
                            delta = data["choices"][0]["message"]["content"]
                            text += delta
                            yield CompletionResponse(delta=delta, text=text, raw=data)

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = self._stream_complete
        return stream_complete_fn(prompt, **kwargs)

    async def _astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        import aiohttp

        url = f"{self.api_base}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self._get_all_kwargs(**kwargs),
        }

        async def gen() -> CompletionResponseAsyncGen:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=self.headers
                ) as response:
                    response.raise_for_status()
                    text = ""
                    async for line in response.content:
                        line_text = line.decode("utf-8").strip()
                        if line_text.startswith("data:"):
                            data = json.loads(line_text[5:])
                            delta = data["choices"][0]["message"]["content"]
                            text += delta
                            yield CompletionResponse(delta=delta, text=text, raw=data)

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, **kwargs)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            **self._get_all_kwargs(**kwargs),
        }

        def gen() -> ChatResponseGen:
            content = ""
            with requests.Session() as session:
                with session.post(
                    url, json=payload, headers=self.headers, stream=True
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(
                        decode_unicode=True
                    ):  # decode lines to Unicode
                        if line.startswith("data:"):
                            data = json.loads(line[5:])
                            delta = data["choices"][0]["delta"]["content"]
                            content += delta
                            message = ChatMessage(
                                role="assistant", content=content, raw=data
                            )
                            yield ChatResponse(message=message, delta=delta, raw=data)

        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self._stream_chat(messages, **kwargs)

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        import aiohttp

        url = f"{self.api_base}/chat/completions"
        message_dicts = to_openai_message_dicts(messages)
        payload = {
            "model": self.model,
            "messages": message_dicts,
            "stream": True,
            **self._get_all_kwargs(**kwargs),
        }

        async def gen() -> ChatResponseAsyncGen:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=self.headers
                ) as response:
                    response.raise_for_status()
                    content = ""
                    async for line in response.content:
                        line_text = line.decode("utf-8").strip()
                        if line_text.startswith("data:"):
                            data = json.loads(line_text[5:])
                            delta = data["choices"][0]["delta"]["content"]
                            content += delta
                            message = ChatMessage(
                                role="assistant", content=content, raw=data
                            )
                            yield ChatResponse(message=message, delta=delta, raw=data)

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        return await self._astream_chat(messages, **kwargs)
