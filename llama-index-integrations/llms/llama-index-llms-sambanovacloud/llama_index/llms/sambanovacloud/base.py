import aiohttp

from typing import Any, Dict, List, Optional, Iterator, Sequence, AsyncIterator

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
from llama_index.core.bridge.pydantic import Field, SecretStr
import json

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
    """
    SambaNova Cloud model.

    Setup:
        To use, you should have the environment variables:
        ``SAMBANOVA_URL`` set with your SambaNova Cloud URL.
        ``SAMBANOVA_API_KEY`` set with your SambaNova Cloud API Key.
        http://cloud.sambanova.ai/

    Example:
        .. code-block:: python
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
        .. code-block:: python

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
            )
    Complete:
        .. code-block:: python
            prompt = "Tell me about Naruto Uzumaki in one sentence"
            response = llm.complete(prompt)

    Chat:
        .. code-block:: python
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
                ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
            ]
            response = llm.chat(messages)

    Stream:
        .. code-block:: python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        for chunk in llm.stream_complete(prompt):
            print(chunk.text)
        for chunk in llm.stream_chat(messages):
            print(chunk.message.content)

    Async:
        .. code-block:: python
        prompt = "Tell me about Naruto Uzumaki in one sentence"
        asyncio.run(llm.acomplete(prompt))

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        asyncio.run(llm.achat(chat_text_msgs))

    Response metadata and usage
        .. code-block:: python

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=("You're a helpful assistant")),
            ChatMessage(role=MessageRole.USER, content="Tell me about Naruto Uzumaki in one sentence")
        ]
        metadata_and_usage = llm.chat(messages).message.additional_kwargs
        print(metadata_and_usage)
    """

    sambanova_url: str = Field(default_factory=str, description="SambaNova Cloud Url")

    sambanova_api_key: SecretStr = Field(
        default_factory=str, description="SambaNova Cloud api key"
    )

    model: str = Field(
        default="Meta-Llama-3.1-8B-Instruct",
        description="The name of the model",
    )

    streaming: bool = Field(
        default=False,
        description="Whether to use streaming handler when using non streaming methods",
    )

    max_tokens: int = Field(default=1024, description="max tokens to generate")

    temperature: float = Field(default=0.7, description="model temperature")

    top_p: Optional[float] = Field(default=None, description="model top p")

    top_k: Optional[int] = Field(default=None, description="model top k")

    stream_options: dict = Field(
        default={"include_usage": True},
        description="stream options, include usage to get generation metrics",
    )

    @classmethod
    def class_name(cls) -> str:
        return "SambaNovaCloud"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=None,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.model,
        )

    def __init__(self, **kwargs: Any) -> None:
        """Init and validate environment variables."""
        kwargs["sambanova_url"] = get_from_param_or_env(
            "url",
            kwargs.get("sambanova_url"),
            "SAMBANOVA_URL",
            default="https://api.sambanova.ai/v1/chat/completions",
        )
        kwargs["sambanova_api_key"] = get_from_param_or_env(
            "api_key", kwargs.get("sambanova_api_key"), "SAMBANOVA_API_KEY"
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
