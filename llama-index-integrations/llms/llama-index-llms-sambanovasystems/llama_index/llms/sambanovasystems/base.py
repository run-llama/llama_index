import aiohttp

from typing import Any, Dict, List, Optional, Iterator, Sequence, AsyncIterator, Tuple

import requests
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
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
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    achat_to_completion_decorator,
)
from llama_index.core.bridge.pydantic import ConfigDict, Field, SecretStr
import json
from requests import Response

from dotenv import load_dotenv

load_dotenv()


def _convert_message_to_dict(message: ChatMessage) -> Dict[str, Any]:
    """Converts a ChatMessage to a dictionary with Role / content.

    Args:
        message: ChatMessage
    Returns:
        messages_dict:  role / content dict
    """
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _create_message_dicts(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """Converts a list of ChatMessages to a list of dictionaries with Role / content.

    Args:
        messages: list of ChatMessages
    Returns:
        messages_dicts:  list of role / content dicts
    """
    return [_convert_message_to_dict(m) for m in messages]


class SambaNovaCloud(LLM):
    """SambaNova Cloud models.

    Setup:
        To use, you should have the environment variables:
        `SAMBANOVA_URL` set with your SambaNova Cloud URL.
        `SAMBANOVA_API_KEY` set with your SambaNova Cloud API Key.
        http://cloud.sambanova.ai/.
        Additionally, download the following packages:
        `pip install llama-index-llms-sambanovasystems`
        `pip install sseclient-py`
    Examples:
    ```python
    SambaNovaCloud(
        sambanova_url = SambaNova cloud endpoint URL,
        sambanova_api_key = set with your SambaNova cloud API key,
        model = model name,
        max_tokens = max number of tokens to generate,
        temperature = model temperature,
        top_p = model top p,
        top_k = model top k,
        stream_options = include usage to get generation metrics
    )
    ```
    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct.
        streaming: bool
            Whether to use streaming handler when using non streaming methods
        max_tokens: int
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        top_k: int
            model top k
        stream_options: dict
            stream options, include usage to get generation metrics
    Key init args — client params:
        sambanova_url: str
            SambaNova Cloud Url
        sambanova_api_key: str
            SambaNova Cloud api key
    Instantiate:
            ```python
            from llama_index.llms.sambanovacloud import SambaNovaCloud
            llm = SambaNovaCloud(
                sambanova_url = SambaNova cloud endpoint URL,
                sambanova_api_key = set with your SambaNova cloud API key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                stream_options = include usage to get generation metrics
                context_window = model context window
            )
            ```
    Complete:
            ```python
            prompt = "Tell me about Naruto Uzumaki in one sentence"
            response = llm.complete(prompt)
            ```
    Chat:
            ```python
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
                ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
            ]
            response = llm.chat(messages)
            ```
    Stream:
        ```python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        for chunk in llm.stream_complete(prompt):
            print(chunk.text)
        for chunk in llm.stream_chat(messages):
            print(chunk.message.content)
        ```
    Async:
        ```python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        asyncio.run(llm.acomplete(prompt))
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        asyncio.run(llm.achat(chat_text_msgs))
        ```
    Response metadata and usage
        ```python
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        metadata_and_usage = llm.chat(messages).message.additional_kwargs
        print(metadata_and_usage)
        ```
    """

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )

    sambanova_url: str = Field(description="SambaNova Cloud Url")

    sambanova_api_key: SecretStr = Field(description="SambaNova Cloud api key")

    model: str = Field(
        default="Meta-Llama-3.1-8B-Instruct",
        description="The name of the model",
    )

    streaming: bool = Field(
        default=False,
        description="Whether to use streaming handler when using non streaming methods",
    )

    context_window: int = Field(default=4096, description="context window")

    max_tokens: int = Field(default=1024, description="max tokens to generate")

    temperature: float = Field(default=0.7, description="model temperature")

    top_p: Optional[float] = Field(default=None, description="model top p")

    top_k: Optional[int] = Field(default=None, description="model top k")

    stream_options: dict = Field(
        default_factory=lambda: {"include_usage": True},
        description="stream options, include usage to get generation metrics",
    )

    @classmethod
    def class_name(cls) -> str:
        return "SambaNovaCloud"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    def __init__(self, **kwargs: Any) -> None:
        """Init and validate environment variables."""
        kwargs["sambanova_url"] = get_from_param_or_env(
            "sambanova_api_key",
            kwargs.get("sambanova_api_key"),
            "SAMBANOVA_URL",
            default="https://api.sambanova.ai/v1/chat/completions",
        )
        kwargs["sambanova_api_key"] = get_from_param_or_env(
            "sambanova_api_key", kwargs.get("sambanova_api_key"), "SAMBANOVA_API_KEY"
        )
        super().__init__(**kwargs)

    def _handle_request(
        self, messages_dicts: List[Dict], stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Performs a post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens
        Returns:
            A response dict.
        """
        data = {
            "messages": messages_dicts,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        http_session = requests.Session()
        response = http_session.post(
            self.sambanova_url,
            headers={
                "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=data,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}.",
                f"{response.text}.",
            )
        response_dict = response.json()
        if response_dict.get("error"):
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}.",
                f"{response_dict}.",
            )
        return response_dict

    async def _handle_request_async(
        self, messages_dicts: List[Dict], stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Performs a async post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens
        Returns:
            A response dict.
        """
        data = {
            "messages": messages_dicts,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.sambanova_url,
                headers={
                    "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=data,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code {response.status}.",
                        f"{await response.text()}.",
                    )
                response_dict = await response.json()
                if response_dict.get("error"):
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code {response.status}.",
                        f"{response_dict}.",
                    )
                return response_dict

    def _handle_streaming_request(
        self, messages_dicts: List[Dict], stop: Optional[List[str]] = None
    ) -> Iterator[Dict]:
        """
        Performs an streaming post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens
        Yields:
            An iterator of response dicts.
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )
        data = {
            "messages": messages_dicts,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": True,
            "stream_options": self.stream_options,
        }
        http_session = requests.Session()
        response = http_session.post(
            self.sambanova_url,
            headers={
                "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=data,
            stream=True,
        )

        client = sseclient.SSEClient(response)

        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )

        for event in client.events():
            if event.event == "error_event":
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}."
                    f"{event.data}."
                )

            try:
                # check if the response is a final event
                # in that case event data response is '[DONE]'
                if event.data != "[DONE]":
                    if isinstance(event.data, str):
                        data = json.loads(event.data)
                    else:
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                    if data.get("error"):
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                    yield data
            except Exception as e:
                raise RuntimeError(
                    f"Error getting content chunk raw streamed response: {e}"
                    f"data: {event.data}"
                )

    async def _handle_streaming_request_async(
        self, messages_dicts: List[Dict], stop: Optional[List[str]] = None
    ) -> AsyncIterator[Dict]:
        """
        Performs an async streaming post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens
        Yields:
            An iterator of response dicts.
        """
        data = {
            "messages": messages_dicts,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": True,
            "stream_options": self.stream_options,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.sambanova_url,
                headers={
                    "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=data,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status}. {await response.text()}"
                    )

                async for line in response.content:
                    if line:
                        event = line.decode("utf-8").strip()

                    if event.startswith("data:"):
                        event = event[len("data:") :].strip()
                        if event == "[DONE]":
                            break
                    elif len(event) == 0:
                        continue

                    try:
                        data = json.loads(event)
                        if data.get("error"):
                            raise RuntimeError(
                                f'Sambanova /complete call failed: {data["error"]}'
                            )
                        yield data
                    except json.JSONDecodeError:
                        raise RuntimeError(
                            f"Sambanova /complete call failed to decode response: {event}"
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error processing response: {e} data: {event}"
                        )

    @llm_chat_callback()
    def chat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Calls the chat implementation of the SambaNovaCloud model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.

        Returns:
            ChatResponse with model generation
        """
        messages_dicts = _create_message_dicts(messages)

        response = self._handle_request(messages_dicts, stop)
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response["choices"][0]["message"]["content"],
            additional_kwargs={
                "id": response["id"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "usage": response.get("usage"),
                "model_name": response["model"],
                "system_fingerprint": response["system_fingerprint"],
                "created": response["created"],
            },
        )
        return ChatResponse(message=message)

    @llm_chat_callback()
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponseGen:
        """
        Streams the chat output of the SambaNovaCloud model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.

        Yields:
            ChatResponseGen with model partial generation
        """
        messages_dicts = _create_message_dicts(messages)

        finish_reason = None
        content = ""
        role = MessageRole.ASSISTANT

        for partial_response in self._handle_streaming_request(messages_dicts, stop):
            if len(partial_response["choices"]) > 0:
                content_delta = partial_response["choices"][0]["delta"]["content"]
                content += content_delta
                additional_kwargs = {
                    "id": partial_response["id"],
                    "finish_reason": partial_response["choices"][0].get(
                        "finish_reason"
                    ),
                }
            else:
                additional_kwargs = {
                    "id": partial_response["id"],
                    "finish_reason": finish_reason,
                    "usage": partial_response.get("usage"),
                    "model_name": partial_response["model"],
                    "system_fingerprint": partial_response["system_fingerprint"],
                    "created": partial_response["created"],
                }

            # yield chunk
            yield ChatResponse(
                message=ChatMessage(
                    role=role, content=content, additional_kwargs=additional_kwargs
                ),
                delta=content_delta,
                raw=partial_response,
            )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    ### Async ###
    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Calls the async chat implementation of the SambaNovaCloud model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.

        Returns:
            ChatResponse with async model generation
        """
        messages_dicts = _create_message_dicts(messages)
        response = await self._handle_request_async(messages_dicts, stop)
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response["choices"][0]["message"]["content"],
            additional_kwargs={
                "id": response["id"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "usage": response.get("usage"),
                "model_name": response["model"],
                "system_fingerprint": response["system_fingerprint"],
                "created": response["created"],
            },
        )
        return ChatResponse(message=message)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "SambaNovaCloud does not currently support async streaming."
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError(
            "SambaNovaCloud does not currently support async streaming."
        )


class SambaStudio(LLM):
    """SambaStudio model.

    Setup:
        To use, you should have the environment variables:
        ``SAMBASTUDIO_URL`` set with your SambaStudio deployed endpoint URL.
        ``SAMBASTUDIO_API_KEY`` set with your SambaStudio deployed endpoint Key.
        https://docs.sambanova.ai/sambastudio/latest/index.html
        Examples:
            ```python
            SambaStudio(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model or expert name (set for CoE endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                context_window = model context window,
                top_p = model top p,
                top_k = model top k,
                do_sample = whether to do sample
                process_prompt = whether to process prompt
                    (set for CoE generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, end special tokens
                    (set for CoE generic v1 and v2 endpoints when process prompt
                     set to false or for StandAlone v1 and v2 endpoints)
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )
            ```
    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for CoE endpoints).
        streaming: bool
            Whether to use streaming
        max_tokens: inthandler when using non streaming methods
            max tokens to generate
        context_window: int
            model context window
        temperature: float
            model temperature
        top_p: float
            model top p
        top_k: int
            model top k
        do_sample: bool
            whether to do sample
        process_prompt:
            whether to process prompt (set for CoE generic v1 and v2 endpoints)
        stream_options: dict
            stream options, include usage to get generation metrics
        special_tokens: dict
            start, start_role, end_role and end special tokens
            (set for CoE generic v1 and v2 endpoints when process prompt set to false
             or for StandAlone v1 and v2 endpoints) default to llama3 special tokens
        model_kwargs: dict
            Extra Key word arguments to pass to the model.
    Key init args — client params:
        sambastudio_url: str
            SambaStudio endpoint URL
        sambastudio_api_key: str
            SambaStudio endpoint api key
    Instantiate:
        ```python
        from llama_index.llms.sambanova import SambaStudio
        llm = SambaStudio=(
            sambastudio_url = set with your SambaStudio deployed endpoint URL,
            sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
            model = model or expert name (set for CoE endpoints),
            max_tokens = max number of tokens to generate,
            temperature = model temperature,
            context_window = model context window,
            top_p = model top p,
            top_k = model top k,
            do_sample = whether to do sample
            process_prompt = whether to process prompt
                (set for CoE generic v1 and v2 endpoints)
            stream_options = include usage to get generation metrics
            special_tokens = start, start_role, end_role, and special tokens
                (set for CoE generic v1 and v2 endpoints when process prompt
                 set to false or for StandAlone v1 and v2 endpoints)
            model_kwargs: Optional = Extra Key word arguments to pass to the model.
        )
        ```
    Complete:
        ```python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        response = llm.complete(prompt)
        ```
    Chat:
        ```python
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        response = llm.chat(messages)
        ```
    Stream:
        ```python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        for chunk in llm.stream_complete(prompt):
            print(chunk.text)
        for chunk in llm.stream_chat(messages):
            print(chunk.message.content)
        ```
    Async:
        ```python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        asyncio.run(llm.acomplete(prompt))
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        asyncio.run(llm.achat(chat_text_msgs))
        ```
    Response metadata and usage
        ```python
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        metadata_and_usage = llm.chat(messages).message.additional_kwargs
        print(metadata_and_usage)
        ```
    """

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )

    sambastudio_url: str = Field(description="SambaStudio Url")

    sambastudio_api_key: SecretStr = Field(description="SambaStudio api key")

    base_url: str = Field(
        default="", exclude=True, description="SambaStudio non streaming Url"
    )

    streaming_url: str = Field(
        default="", exclude=True, description="SambaStudio streaming Url"
    )

    model: Optional[str] = Field(
        default=None,
        description="The name of the model or expert to use (for CoE endpoints)",
    )

    streaming: bool = Field(
        default=False,
        description="Whether to use streaming handler when using non streaming methods",
    )

    context_window: int = Field(default=4096, description="context window")

    max_tokens: int = Field(default=1024, description="max tokens to generate")

    temperature: Optional[float] = Field(default=0.7, description="model temperature")

    top_p: Optional[float] = Field(default=None, description="model top p")

    top_k: Optional[int] = Field(default=None, description="model top k")

    do_sample: Optional[bool] = Field(
        default=None, description="whether to do sampling"
    )

    process_prompt: Optional[bool] = Field(
        default=True,
        description="whether process prompt (for CoE generic v1 and v2 endpoints)",
    )

    stream_options: dict = Field(
        default_factory=lambda: {"include_usage": True},
        description="stream options, include usage to get generation metrics",
    )

    special_tokens: dict = Field(
        default={
            "start": "<|begin_of_text|>",
            "start_role": "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>",
            "end_role": "<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>\n",
        },
        description="start, start_role, end_role and end special tokens (set for CoE generic v1 and v2 endpoints when process prompt set to false or for StandAlone v1 and v2 endpoints) default to llama3 special tokens",
    )

    model_kwargs: Optional[Dict[str, Any]] = Field(
        default=None, description="Key word arguments to pass to the model."
    )

    @classmethod
    def class_name(cls) -> str:
        return "SambaStudio"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    def __init__(self, **kwargs: Any) -> None:
        """Init and validate environment variables."""
        kwargs["sambastudio_url"] = get_from_param_or_env(
            "sambastudio_url", kwargs.get("sambastudio_url"), "SAMBASTUDIO_URL"
        )
        kwargs["sambastudio_api_key"] = get_from_param_or_env(
            "sambastudio_api_key",
            kwargs.get("sambastudio_api_key"),
            "SAMBASTUDIO_API_KEY",
        )
        kwargs["sambastudio_url"], kwargs["streaming_url"] = self._get_sambastudio_urls(
            kwargs["sambastudio_url"]
        )
        super().__init__(**kwargs)

    def _messages_to_string(self, messages: Sequence[ChatMessage]) -> str:
        """Convert a sequence of ChatMessages to:
        - dumped json string with Role / content dict structure when process_prompt is true,
        - string with special tokens if process_prompt is false for generic V1 and V2 endpoints.

        Args:
            messages: sequence of ChatMessages
        Returns:
            str: string to send as model input depending on process_prompt param
        """
        if self.process_prompt:
            messages_dict: Dict[str, Any] = {
                "conversation_id": "sambaverse-conversation-id",
                "messages": [],
            }
            for message in messages:
                messages_dict["messages"].append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )
            messages_string = json.dumps(messages_dict)
        else:
            messages_string = self.special_tokens["start"]
            for message in messages:
                messages_string += self.special_tokens["start_role"].format(
                    role=self._get_role(message)
                )
                messages_string += f" {message.content} "
                messages_string += self.special_tokens["end_role"]
            messages_string += self.special_tokens["end"]

        return messages_string

    def _get_sambastudio_urls(self, url: str) -> Tuple[str, str]:
        """Get streaming and non streaming URLs from the given URL.

        Args:
            url: string with sambastudio base or streaming endpoint url
        Returns:
            base_url: string with url to do non streaming calls
            streaming_url: string with url to do streaming calls
        """
        if "chat/completions" in url:
            base_url = url
            stream_url = url
        else:
            if "stream" in url:
                base_url = url.replace("stream/", "")
                stream_url = url
            else:
                base_url = url
                if "generic" in url:
                    stream_url = "generic/stream".join(url.split("generic"))
                else:
                    raise ValueError("Unsupported URL")
        return base_url, stream_url

    def _handle_request(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = False,
    ) -> Response:
        """Performs a post request to the LLM API.

        Args:
        messages_dicts: List of role / content dicts to use as input.
        stop: list of stop tokens
        streaming: whether to do a streaming call
        Returns:
            A request Response object
        """
        # create request payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            messages_dicts = _create_message_dicts(messages)
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stream": streaming,
                "stream_options": self.stream_options,
            }
            data = {key: value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer "
                f"{self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

        # create request payload for generic v1 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            items = [{"id": "item0", "value": self._messages_to_string(messages)}]
            params: Dict[str, Any] = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
            }
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {key: value for key, value in params.items() if value is not None}
            data = {"items": items, "params": params}
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        # create request payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            params = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
            }
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {
                key: {"type": type(value).__name__, "value": str(value)}
                for key, value in params.items()
                if value is not None
            }
            if streaming:
                data = {
                    "instance": self._messages_to_string(messages),
                    "params": params,
                }
            else:
                data = {
                    "instances": [self._messages_to_string(messages)],
                    "params": params,
                }
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        http_session = requests.Session()
        if streaming:
            response = http_session.post(
                self.streaming_url, headers=headers, json=data, stream=True
            )
        else:
            response = http_session.post(
                self.base_url, headers=headers, json=data, stream=False
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )
        return response

    async def _handle_request_async(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = False,
    ) -> Response:
        """Performs an async post request to the LLM API.

        Args:
        messages_dicts: List of role / content dicts to use as input.
        stop: list of stop tokens
        streaming: whether to do a streaming call
        Returns:
            A request Response object
        """
        # create request payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            messages_dicts = _create_message_dicts(messages)
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stream": streaming,
                "stream_options": self.stream_options,
            }
            data = {key: value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer "
                f"{self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

        # create request payload for generic v1 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            items = [{"id": "item0", "value": self._messages_to_string(messages)}]
            params: Dict[str, Any] = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
            }
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {key: value for key, value in params.items() if value is not None}
            data = {"items": items, "params": params}
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        # create request payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            params = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
            }
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {
                key: {"type": type(value).__name__, "value": str(value)}
                for key, value in params.items()
                if value is not None
            }
            if streaming:
                data = {
                    "instance": self._messages_to_string(messages),
                    "params": params,
                }
            else:
                data = {
                    "instances": [self._messages_to_string(messages)],
                    "params": params,
                }
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        async with aiohttp.ClientSession() as session:
            if streaming:
                url = self.streaming_url
            else:
                url = self.base_url

            async with session.post(
                url,
                headers=headers,
                json=data,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status}."
                        f"{response.text}."
                    )
                response_dict = await response.json()
                if response_dict.get("error"):
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code {response.status}.",
                        f"{response_dict}.",
                    )
                return response_dict

    def _process_response(self, response: Response) -> ChatMessage:
        """Process a non streaming response from the api.

        Args:
            response: A request Response object
        Returns:
            generation: a ChatMessage with model generation
        """
        # Extract json payload form response
        try:
            response_dict = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )

        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            content = response_dict["choices"][0]["message"]["content"]
            response_metadata = {
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            }

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            content = response_dict["items"][0]["value"]["completion"]
            response_metadata = response_dict["items"][0]

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            content = response_dict["predictions"][0]["completion"]
            response_metadata = response_dict

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        return ChatMessage(
            content=content,
            additional_kwargs=response_metadata,
            role=MessageRole.ASSISTANT,
        )

    def _process_stream_response(self, response: Response) -> Iterator[ChatMessage]:
        """Process a streaming response from the api.

        Args:
            response: An iterable request Response object
        Yields:
            generation: an Iterator[ChatMessage] with model partial generation
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )

        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            finish_reason = ""
            content = ""
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.event == "error_event":
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}."
                        f"{event.data}."
                    )
                try:
                    # check if the response is not a final event ("[DONE]")
                    if event.data != "[DONE]":
                        if isinstance(event.data, str):
                            data = json.loads(event.data)
                        else:
                            raise RuntimeError(
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                        if data.get("error"):
                            raise RuntimeError(
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                        if len(data["choices"]) > 0:
                            finish_reason = data["choices"][0].get("finish_reason")
                            content += data["choices"][0]["delta"]["content"]
                            id = data["id"]
                            metadata = {}
                        else:
                            content += ""
                            id = data["id"]
                            metadata = {
                                "finish_reason": finish_reason,
                                "usage": data.get("usage"),
                                "model_name": data["model"],
                                "system_fingerprint": data["system_fingerprint"],
                                "created": data["created"],
                            }
                        if data.get("usage") is not None:
                            content += ""
                            id = data["id"]
                            metadata = {
                                "finish_reason": finish_reason,
                                "usage": data.get("usage"),
                                "model_name": data["model"],
                                "system_fingerprint": data["system_fingerprint"],
                                "created": data["created"],
                            }
                        yield ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=content,
                            additional_kwargs=metadata,
                        )

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"data: {event.data}"
                    )

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            content = ""
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content += data["result"]["items"][0]["value"]["stream_token"]
                    id = data["result"]["items"][0]["id"]
                    if data["result"]["items"][0]["value"]["is_last_response"]:
                        metadata = {
                            "finish_reason": data["result"]["items"][0]["value"].get(
                                "stop_reason"
                            ),
                            "prompt": data["result"]["items"][0]["value"].get("prompt"),
                            "usage": {
                                "prompt_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("prompt_tokens_count"),
                                "completion_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("completion_tokens_count"),
                                "total_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("total_tokens_count"),
                                "start_time": data["result"]["items"][0]["value"].get(
                                    "start_time"
                                ),
                                "end_time": data["result"]["items"][0]["value"].get(
                                    "end_time"
                                ),
                                "model_execution_time": data["result"]["items"][0][
                                    "value"
                                ].get("model_execution_time"),
                                "time_to_first_token": data["result"]["items"][0][
                                    "value"
                                ].get("time_to_first_token"),
                                "throughput_after_first_token": data["result"]["items"][
                                    0
                                ]["value"].get("throughput_after_first_token"),
                                "batch_size_used": data["result"]["items"][0][
                                    "value"
                                ].get("batch_size_used"),
                            },
                        }
                    else:
                        metadata = {}
                    yield ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        additional_kwargs=metadata,
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            content = ""
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content += data["result"]["responses"][0]["stream_token"]
                    id = None
                    if data["result"]["responses"][0]["is_last_response"]:
                        metadata = {
                            "finish_reason": data["result"]["responses"][0].get(
                                "stop_reason"
                            ),
                            "prompt": data["result"]["responses"][0].get("prompt"),
                            "usage": {
                                "prompt_tokens_count": data["result"]["responses"][
                                    0
                                ].get("prompt_tokens_count"),
                                "completion_tokens_count": data["result"]["responses"][
                                    0
                                ].get("completion_tokens_count"),
                                "total_tokens_count": data["result"]["responses"][
                                    0
                                ].get("total_tokens_count"),
                                "start_time": data["result"]["responses"][0].get(
                                    "start_time"
                                ),
                                "end_time": data["result"]["responses"][0].get(
                                    "end_time"
                                ),
                                "model_execution_time": data["result"]["responses"][
                                    0
                                ].get("model_execution_time"),
                                "time_to_first_token": data["result"]["responses"][
                                    0
                                ].get("time_to_first_token"),
                                "throughput_after_first_token": data["result"][
                                    "responses"
                                ][0].get("throughput_after_first_token"),
                                "batch_size_used": data["result"]["responses"][0].get(
                                    "batch_size_used"
                                ),
                            },
                        }
                    else:
                        metadata = {}
                    yield ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        additional_kwargs=metadata,
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

    async def _process_response_async(
        self, response_dict: Dict[str, Any]
    ) -> ChatMessage:
        """Process a non streaming response from the api.

        Args:
            response: A request Response object
        Returns:
            generation: a ChatMessage with model generation
        """
        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            content = response_dict["choices"][0]["message"]["content"]
            response_metadata = {
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            }

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            content = response_dict["items"][0]["value"]["completion"]
            response_metadata = response_dict["items"][0]

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            content = response_dict["predictions"][0]["completion"]
            response_metadata = response_dict

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        return ChatMessage(
            content=content,
            additional_kwargs=response_metadata,
            role=MessageRole.ASSISTANT,
        )

    @llm_chat_callback()
    def chat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Calls the chat implementation of the SambaStudio model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.

        Returns:
            ChatResponse with model generation
        """
        # if self.streaming:
        #     stream_iter = self._stream(
        #         messages, stop=stop, **kwargs
        #     )
        #     if stream_iter:
        #         return generate_from_stream(stream_iter)
        response = self._handle_request(messages, stop, streaming=False)
        message = self._process_response(response)

        return ChatResponse(message=message)

    @llm_chat_callback()
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream the output of the SambaStudio model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.

        Yields:
            chunk: ChatResponseGen with model partial generation
        """
        response = self._handle_request(messages, stop, streaming=True)
        for ai_message_chunk in self._process_stream_response(response):
            chunk = ChatResponse(message=ai_message_chunk)
            yield chunk

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Calls the chat implementation of the SambaStudio model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.

        Returns:
            ChatResponse with model generation
        """
        response_dict = await self._handle_request_async(
            messages, stop, streaming=False
        )
        message = await self._process_response_async(response_dict)
        return ChatResponse(message=message)

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError(
            "SambaStudio does not currently support async streaming."
        )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError(
            "SambaStudio does not currently support async streaming."
        )
