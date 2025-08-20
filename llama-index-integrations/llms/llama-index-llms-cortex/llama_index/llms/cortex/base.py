import json
import os
from typing import Any, Dict, Optional, Sequence
import warnings

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

from llama_index.core.callbacks import CallbackManager
from llama_index.llms.cortex.utils import (
    generate_sf_jwt,
    is_spcs_environment,
    get_default_spcs_token,
    get_spcs_base_url,
)

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_TOKENS = 4_096  # !! This is the max for all requests to the Rest API, per # https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api
DEFAULT_MODEL = "llama3.2-1b"
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0

# https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#model-restrictions
# https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api
model_specs = {
    "claude-3-5-sonnet": {"context_window": 200_000, "max_output": None},
    "llama4-maverick": {"context_window": 128_000, "max_output": None},
    "llama3.2-1b": {"context_window": 128_000, "max_output": None},
    "llama3.2-3b": {"context_window": 128_000, "max_output": None},
    "llama3.1-8b": {"context_window": 128_000, "max_output": None},
    "llama3.1-70b": {"context_window": 128_000, "max_output": None},
    "llama3.3-70b": {"context_window": 128_000, "max_output": None},
    "snowflake-llama-3.3-70b": {"context_window": 8_000, "max_output": None},
    "llama3.1-405b": {"context_window": 128_000, "max_output": None},
    "snowflake-llama-3.1-405b": {"context_window": 8_000, "max_output": None},
    "snowflake-arctic": {"context_window": 4_096, "max_output": None},
    "deepseek-r1": {"context_window": 32_768, "max_output": None},
    "reka-core": {"context_window": 32_000, "max_output": None},
    "reka-flash": {"context_window": 100_000, "max_output": None},
    "mistral-large": {"context_window": 32_000, "max_output": None},
    "mistral-large2": {"context_window": 128_000, "max_output": None},
    "mixtral-8x7b": {"context_window": 32000, "max_output": None},
    "mistral-7b": {"context_window": 32000, "max_output": None},
    "jamba-instruct": {"context_window": 256_000, "max_output": None},
    "jamba-1.5-mini": {"context_window": 256_000, "max_output": None},
    "jamba-1.5-large": {"context_window": 256_000, "max_output": None},
    "gemma-7b": {"context_window": 8_000, "max_output": None},
    "llama2-70b-chat": {"context_window": 4_096, "max_output": None},
    "llama3-8b": {"context_window": 8_000, "max_output": None},
    "llama3-70b": {"context_window": 8_000, "max_output": None},
}


class Cortex(CustomLLM):
    """
    Cortex LLM.

    This class provides an interface to Snowflake's Cortex LLM service.
    HTTP errors from the API (including invalid model names) will raise
    requests.exceptions.HTTPError for synchronous methods or
    aiohttp.ClientResponseError for asynchronous methods.

    This class provides an interface to Snowflake's Cortex LLM service.
    HTTP errors from the API (including invalid model names) will raise
    requests.exceptions.HTTPError for synchronous methods or
    aiohttp.ClientResponseError for asynchronous methods.

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
        default=None,
        description="The maximum number of context tokens for the model.",
    )
    max_tokens: int = Field(
        default=None,
        description="The maximum number of tokens to generate in response.",
    )
    model: str = Field(default=DEFAULT_MODEL, description="The model to use.")

    jwt_token: str = Field(default=None, description="JWT token data or filepath")
    session: Optional[Any] = Field(default=None, description="Snowpark Session object.")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        user: Optional[str] = None,
        account: Optional[str] = None,
        private_key_file: Optional[str] = None,
        jwt_token: Optional[str] = None,
        session: Optional[Any] = None,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Implements all Snowflake Cortex LLMs.

        AUTHENTICATION:
        The recommended way to connect is to use a snowflake.snowpark.Session object.
        Environment variables SNOWFLAKE_ACCOUNT and SNOWFLAKE_USERNAME must be set or passed as params.

        Authentication methods (in order of precedence):
        1. Explicitly provided authentication (one of: private_key_file, jwt_token, or session)
        2. Environment variable SNOWFLAKE_KEY_FILE (legacy support)
        3. Default OAUTH token in SPCS environment

        If none of these methods are available, an assertion error is raised.

        """
        super().__init__(
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
        )

        # Set model specs first
        self.model = model
        specs = model_specs.get(self.model, {})
        self.context_window = specs.get("context_window") or DEFAULT_CONTEXT_WINDOW
        self.max_tokens = specs.get("max_output") or DEFAULT_MAX_TOKENS

        # Set user and account
        self.user = user or os.environ.get("SNOWFLAKE_USERNAME")
        self.account = account or os.environ.get("SNOWFLAKE_ACCOUNT")

        # Authentication logic - prioritize explicit authentication methods
        is_in_spcs = is_spcs_environment()
        env_key_file = os.environ.get("SNOWFLAKE_KEY_FILE")

        # Case 1: Explicit authentication provided
        if private_key_file:
            self.private_key_file = private_key_file
        elif jwt_token:
            if os.path.isfile(jwt_token):
                with open(jwt_token) as fp:
                    self.jwt_token = fp.read()
            else:
                self.jwt_token = jwt_token
        elif session:
            self.session = session

        # Case 2: Environment variable private key (legacy behavior)
        elif env_key_file and not is_in_spcs:
            self.private_key_file = env_key_file

        # Case 3: SPCS environment with default token
        elif is_in_spcs:
            self.jwt_token = get_default_spcs_token()

        # No valid authentication method found
        else:
            raise ValueError(
                "Authentication required. Please provide one of: private_key_file, jwt_token, session, "
                "set SNOWFLAKE_KEY_FILE environment variable, or run in SPCS environment."
            )
        if is_in_spcs and self.session:
            warnings.warn(
                "SPCS environment detected. If using the default auth token, please ensure you do NOT set values for 'user' and 'role' parameters or your auth may be rejected."
            )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
            is_function_calling_model=False,
        )

    @property
    def snowflake_api_endpoint(self) -> str:
        if is_spcs_environment():
            return get_spcs_base_url()
        else:
            base_url = f"https://{self.account}.snowflakecomputing.com"
        return base_url

    @property
    def cortex_complete_endpoint(self) -> str:
        append = "/api/v2/cortex/inference:complete"
        return self.snowflake_api_endpoint + append

    @property
    def snowflake_api_endpoint(self) -> str:
        if is_spcs_environment():
            return "https://" + get_spcs_base_url()
        else:
            base_url = f"https://{self.account}.snowflakecomputing.com"
        return base_url

    @property
    def cortex_complete_endpoint(self) -> str:
        append = "/api/v2/cortex/inference:complete"
        return self.snowflake_api_endpoint + append

    def _make_completion_payload(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> dict:
        """Create a payload for the completions."""
        temperature = kwargs.pop("temperature", DEFAULT_TEMP)
        top_p = kwargs.pop("top_p", DEFAULT_TOP_P)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if not formatted:
            prompt = prompt.format(**kwargs)
        return {
            "url": self.cortex_complete_endpoint,
            "url": self.cortex_complete_endpoint,
            "headers": {
                "Authorization": self._generate_auth_header(),
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
        api_response.raise_for_status()
        api_response.raise_for_status()
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
            await api_response.raise_for_status()
            await api_response.raise_for_status()
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
        api_response.raise_for_status()
        api_response.raise_for_status()

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
        async def gen() -> CompletionResponseAsyncGen:
            async with aiohttp.ClientSession() as session:
                api_response = await session.post(
                    **self._make_completion_payload(prompt, formatted, **kwargs)
                )
                text = ""
                async for line in api_response.content:
                    line = line.decode()
                    if line and (line != "\n") and line.startswith("data: "):
                        line_json = json.loads(line[len("data: ") :].strip("\n"))
                        line_delta = line_json["choices"][0]["delta"].get("content", "")
                        text += line_delta
                        yield CompletionResponse(
                            text=text, delta=line_delta, raw=line_json
                        )

        async def gen() -> CompletionResponseAsyncGen:
            async with aiohttp.ClientSession() as session:
                api_response = await session.post(
                    **self._make_completion_payload(prompt, formatted, **kwargs)
                )
                text = ""
                async for line in api_response.content:
                    line = line.decode()
                    if line and (line != "\n") and line.startswith("data: "):
                        line_json = json.loads(line[len("data: ") :].strip("\n"))
                        line_delta = line_json["choices"][0]["delta"].get("content", "")
                        text += line_delta
                        yield CompletionResponse(
                            text=text, delta=line_delta, raw=line_json
                        )

        return gen()

    def _generate_auth_header(self) -> str:
        # private key file has to be checked 2nd to last,
        # it can be set merely due to an env variable existing
        if self.jwt_token:
            return f"Bearer {self.jwt_token}"
        elif self.session:
            return f'Snowflake Token="{self.session.connection.rest.token}"'
        elif self.private_key_file:
            return f"Bearer {generate_sf_jwt(self.account, self.user, self.private_key_file)}"
        else:
            raise ValueError(
                "llama-index Cortex LLM Error: No authentication method set."
            )

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
        jwt = self._generate_auth_header()
        return {
            "url": self.cortex_complete_endpoint,
            "url": self.cortex_complete_endpoint,
            "headers": {
                "Authorization": self._generate_auth_header(),
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
        api_response.raise_for_status()
        api_response.raise_for_status()
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
            await api_response.raise_for_status()
            await api_response.raise_for_status()
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
        api_response.raise_for_status()
        api_response.raise_for_status()

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
            await api_response.raise_for_status()
            await api_response.raise_for_status()
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
