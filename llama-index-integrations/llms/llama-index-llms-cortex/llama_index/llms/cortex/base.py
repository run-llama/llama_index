import json
import os
from typing import Any, Dict, Optional, Sequence

import aiohttp
import requests
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import (
    CallbackManager,
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.llms.cortex.utils import generate_sf_jwt

DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL = "llama3.2-1b"
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0


class Cortex(CustomLLM):
    """
    Cortex LLM.

    Examples:
        `pip install llama-index-llms-cortex`

        ```python
        from llama_index.llms.cortex import Cortex


        llm = Cortex(
            model="llama3.2-1b",
            user=your_sf_user,
            account=your_sf_account,
            private_key_file=your_sf_private_key_file
        )

        completion_response = llm.complete(
            "write me a haiku about a snowflake",
            temperature=0.0
        )
        print(completion_response)
        ```
    """

    user: str = Field(
        description="Snowflake user.",
        default=os.environ.get("SNOWFLAKE_USERNAME", None),
    )
    account: str = Field(
        description="Fully qualified snowflake account specified as <ORG_ID>-<ACCOUNT_ID>.",
        default=os.environ.get("SNOWFLAKE_ACCOUNT", None),
    )
    private_key_file: str = Field(
        description="Filepath to snowflake private key file.",
        default=os.environ.get("SNOWFLAKE_KEY_FILE", None),
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate in response.",
    )
    model: str = Field(default=DEFAULT_MODEL, description="The model to use.")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        user: Optional[str] = None,
        account: Optional[str] = None,
        private_key_file: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
        )
        self.model = model
        self.user = user or os.environ.get("SNOWFLAKE_USERNAME", None)
        self.account = account or os.environ.get("SNOWFLAKE_ACCOUNT", None)
        self.private_key_file = private_key_file or os.environ.get(
            "SNOWFLAKE_KEY_FILE", None
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            # NOTE: no method exists to get model
            # context window/max output tokens size
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=False,
        )

    @property
    def api_base(self) -> str:
        return f"https://{self.account}.snowflakecomputing.com/api/v2/cortex/inference:complete"

    def _make_completion_payload(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> dict:
        """Create a payload for the completions."""
        temperature = kwargs.pop("temperature", DEFAULT_TEMP)
        top_p = kwargs.pop("top_p", DEFAULT_TOP_P)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if not formatted:
            prompt = prompt.format(**kwargs)
        jwt = generate_sf_jwt(self.account, self.user, self.private_key_file)
        return {
            "url": self.api_base,
            "headers": {
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
                "Authorization": f"Bearer {jwt}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            "json": {
                "model": self.model,
                "messages": [{"content": prompt}],
                "top_p": top_p,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

    def _complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        api_response = requests.post(
            **self._make_completion_payload(prompt, formatted, **kwargs), stream=True
        )
        responses = []
        for line in api_response.iter_lines(decode_unicode=True):
            if line:
                responses.append(json.loads(line[len("data: ") :]))
        return CompletionResponse(
            text="".join(r["choices"][0]["delta"].get("content", "") for r in responses)
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, formatted, **kwargs)

    async def _acomplete(self, prompt, formatted=False, **kwargs) -> CompletionResponse:
        async with aiohttp.ClientSession() as session:
            api_response = await session.post(
                **self._make_completion_payload(prompt, formatted, **kwargs)
            )
            responses = []
            async for line in api_response.content:
                line = line.decode()
                if line and (line != "\n"):
                    x = line.strip()[len("data: ") :].strip("\n")
                    responses.append(json.loads(x))
            return CompletionResponse(
                text="".join(
                    r["choices"][0]["delta"].get("content", "") for r in responses
                )
            )

    @llm_completion_callback()
    async def acomplete(self, prompt, formatted=False, **kwargs) -> CompletionResponse:
        return await self._acomplete(prompt, formatted, **kwargs)

    def _stream_complete(
        self, prompt, formatted=False, **kwargs
    ) -> CompletionResponseGen:
        api_response = requests.post(
            **self._make_completion_payload(prompt, formatted, **kwargs), stream=True
        )

        def gen() -> CompletionResponseGen:
            text = ""
            for line in api_response.iter_lines():
                if line:
                    line_json = json.loads(line[len("data: ") :])
                    line_delta = line_json["choices"][0]["delta"].get("content", "")
                    text += line_delta
                    yield CompletionResponse(text=text, delta=line_delta, raw=line_json)

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt, formatted=False, **kwargs
    ) -> CompletionResponseGen:
        return self._stream_complete(prompt, formatted, **kwargs)

    async def _astream_complete(
        self, prompt, formatted=False, **kwargs
    ) -> CompletionResponseAsyncGen:
        async with aiohttp.ClientSession() as session:
            api_response = await session.post(
                **self._make_completion_payload(prompt, formatted, **kwargs)
            )
            # buffer data
            lines = []
            async for line in api_response.content:
                line = line.decode()
                if line and (line != "\n"):
                    lines.append(line)

        async def gen() -> CompletionResponseAsyncGen:
            text = ""
            for line in lines:
                line_json = json.loads(line[len("data: ") :].strip("\n"))
                line_delta = line_json["choices"][0]["delta"].get("content", "")
                text += line_delta
                yield CompletionResponse(text=text, delta=line_delta, raw=line_json)

        return gen()

    @llm_completion_callback()
    async def astream_complete(
        self, prompt, formatted=False, **kwargs
    ) -> CompletionResponseAsyncGen:
        return await self._astream_complete(prompt, formatted, **kwargs)

    def _make_chat_payload(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> dict:
        """Create a payload for a chat."""
        temperature = kwargs.pop("temperature", DEFAULT_TEMP)
        top_p = kwargs.pop("top_p", DEFAULT_TOP_P)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        jwt = generate_sf_jwt(self.account, self.user, self.private_key_file)
        return {
            "url": self.api_base,
            "headers": {
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
                "Authorization": f"Bearer {jwt}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            "json": {
                "model": self.model,
                "messages": [
                    {"role": message.role.lower(), "content": message.content}
                    for message in messages
                ],
                "top_p": top_p,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        api_response = requests.post(
            **self._make_chat_payload(messages, **kwargs), stream=True
        )
        responses = []
        for line in api_response.iter_lines(decode_unicode=True):
            if line:
                responses.append(json.loads(line[len("data: ") :]))
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="".join(
                    r["choices"][0]["delta"].get("content", "") for r in responses
                ),
            ),
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat(messages, **kwargs)

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        async with aiohttp.ClientSession() as session:
            api_response = await session.post(
                **self._make_chat_payload(messages, **kwargs)
            )
            responses = []
            async for line in api_response.content:
                line = line.decode()
                if line and (line != "\n"):
                    responses.append(json.loads(line[len("data: ") :].strip("\n")))
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="".join(
                        r["choices"][0]["delta"].get("content", "") for r in responses
                    ),
                ),
            )

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return await self._achat(messages, **kwargs)

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        api_response = requests.post(
            **self._make_chat_payload(messages, **kwargs), stream=True
        )

        def gen() -> ChatResponseGen:
            text = ""
            for line in api_response.iter_lines():
                if line:
                    line_json = json.loads(line[len("data: ") :])
                    line_delta = line_json["choices"][0]["delta"].get("content", "")
                    text += line_delta
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                        delta=line_delta,
                        raw=line_json,
                    )

        return gen()

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self._stream_chat(messages, **kwargs)

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async with aiohttp.ClientSession() as session:
            api_response = await session.post(
                **self._make_chat_payload(messages, **kwargs)
            )
            # buffer data
            lines = []
            async for line in api_response.content:
                line = line.decode()
                if line and (line != "\n"):
                    lines.append(line)

        async def gen() -> ChatResponseAsyncGen:
            text = ""
            for line in lines:
                line_json = json.loads(line[len("data: ") :].strip("\n"))
                line_delta = line_json["choices"][0]["delta"].get("content", "")
                text += line_delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                    delta=line_delta,
                    raw=line_json,
                )

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        return await self._astream_chat(messages, **kwargs)
