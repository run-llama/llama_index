from typing import Any, Sequence, Any, Dict, List, Union
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
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)

from requests import Response
from requests.exceptions import HTTPError
from http import HTTPStatus

import requests
import json
import os


class MaritalkHTTPError(HTTPError):
    def __init__(self, request_obj: Response) -> None:
        self.request_obj = request_obj
        try:
            response_json = request_obj.json()
            if "detail" in response_json:
                api_message = response_json["detail"]
            elif "message" in response_json:
                api_message = response_json["message"]
            else:
                api_message = response_json
        except Exception:
            api_message = request_obj.text

        self.message = api_message
        self.status_code = request_obj.status_code

    def __str__(self) -> str:
        status_code_meaning = HTTPStatus(self.status_code).phrase
        formatted_message = f"HTTP Error: {self.status_code} - {status_code_meaning}"
        formatted_message += f"\nDetail: {self.message}"
        return formatted_message


class Maritalk(LLM):
    """
    Maritalk LLM.

    Examples:
        `pip install llama-index-llms-maritalk`

        ```python
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.maritalk import Maritalk

        # To customize your API key, do this
        # otherwise it will lookup MARITALK_API_KEY from your env variable
        # llm = Maritalk(api_key="<your_maritalk_api_key>")

        llm = Maritalk()

        # Call chat with a list of messages
        messages = [
            ChatMessage(
                role="system",
                content="You are an assistant specialized in suggesting pet names. Given the animal, you must suggest 4 names.",
            ),
            ChatMessage(role="user", content="I have a dog."),
        ]

        response = llm.chat(messages)
        print(response)
        ```

    """

    api_key: str = Field(
        default=None,
        description="Your MariTalk API key.",
    )

    model: str = Field(
        default="sabia-2-medium",
        description="Chose one of the available models:\n"
        "- `sabia-2-medium`\n"
        "- `sabia-2-small`\n"
        "- `maritalk-2024-01-08`",
    )

    temperature: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description="Run inference with this temperature. Must be in the"
        "closed interval [0.0, 1.0].",
    )

    max_tokens: int = Field(
        default=512,
        gt=0,
        description="The maximum number of tokens togenerate in the reply.",
    )

    do_sample: bool = Field(
        default=True,
        description="Whether or not to use sampling; use `True` to enable.",
    )

    top_p: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Nucleus sampling parameter controlling the size of"
        " the probability mass considered for sampling.",
    )

    _endpoint: str = PrivateAttr("https://chat.maritaca.ai/api/chat/inference")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # If an API key is not provided during instantiation,
        # fall back to the MARITALK_API_KEY environment variable
        self.api_key = self.api_key or os.getenv("MARITALK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "An API key must be provided or set in the "
                "'MARITALK_API_KEY' environment variable."
            )

    @classmethod
    def class_name(cls) -> str:
        return "Maritalk"

    def parse_messages_for_model(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict[str, Union[str, List[Union[str, Dict[Any, Any]]]]]]:
        """
        Parses messages from LlamaIndex's format to the format expected by
        the MariTalk API.

        Parameters
        ----------
            messages (Sequence[ChatMessage]): A list of messages in LlamaIndex
            format to be parsed.

        Returns
        -------
            A list of messages formatted for the MariTalk API.

        """
        formatted_messages = []

        for message in messages:
            if message.role.value == MessageRole.USER:
                role = "user"
            elif message.role.value == MessageRole.ASSISTANT:
                role = "assistant"
            elif message.role.value == MessageRole.SYSTEM:
                role = "system"

            formatted_messages.append({"role": role, "content": message.content})
        return formatted_messages

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="maritalk",
            context_window=self.max_tokens,
            is_chat_model=True,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Prepare the data payload for the Maritalk API
        formatted_messages = self.parse_messages_for_model(messages)

        data = {
            "model": self.model,
            "messages": formatted_messages,
            "do_sample": self.do_sample,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **kwargs,
        }

        headers = {"authorization": f"Key {self.api_key}"}

        response = requests.post(self._endpoint, json=data, headers=headers)

        if response.ok:
            answer = response.json().get("answer", "No answer found")
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                raw=response.json(),
            )
        else:
            raise MaritalkHTTPError(response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        try:
            import httpx

            # Prepare the data payload for the Maritalk API
            formatted_messages = self.parse_messages_for_model(messages)

            data = {
                "model": self.model,
                "messages": formatted_messages,
                "do_sample": self.do_sample,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **kwargs,
            }

            headers = {"authorization": f"Key {self.api_key}"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._endpoint, json=data, headers=headers, timeout=None
                )

            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found")
                return ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                    raw=response.json(),
                )
            else:
                raise MaritalkHTTPError(response)

        except ImportError:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install httpx`."
            )

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        acomplete_fn = achat_to_completion_decorator(self.achat)
        return await acomplete_fn(prompt, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # Prepare the data payload for the Maritalk API
        formatted_messages = self.parse_messages_for_model(messages)

        data = {
            "model": self.model,
            "messages": formatted_messages,
            "do_sample": self.do_sample,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
            **kwargs,
        }

        headers = {"authorization": f"Key {self.api_key}"}

        def gen() -> ChatResponseGen:
            response = requests.post(
                self._endpoint, json=data, headers=headers, stream=True
            )
            if response.ok:
                content = ""
                for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        response_data = line.replace(b"data: ", b"").decode("utf-8")
                        if response_data:
                            parsed_data = json.loads(response_data)
                            if "text" in parsed_data:
                                content_delta = parsed_data["text"]
                                content += content_delta
                                yield ChatResponse(
                                    message=ChatMessage(
                                        role=MessageRole.ASSISTANT, content=content
                                    ),
                                    delta=content_delta,
                                    raw=parsed_data,
                                )
            else:
                raise MaritalkHTTPError(response)

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return stream_complete_fn(prompt, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        try:
            import httpx

            # Prepare the data payload for the Maritalk API
            formatted_messages = self.parse_messages_for_model(messages)

            data = {
                "model": self.model,
                "messages": formatted_messages,
                "do_sample": self.do_sample,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": True,
                **kwargs,
            }

            headers = {"authorization": f"Key {self.api_key}"}

            async def gen() -> ChatResponseAsyncGen:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        self._endpoint,
                        data=json.dumps(data),
                        headers=headers,
                        timeout=None,
                    ) as response:
                        if response.status_code == 200:
                            content = ""
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    response_data = line.replace("data: ", "")
                                    if response_data:
                                        parsed_data = json.loads(response_data)
                                        if "text" in parsed_data:
                                            content_delta = parsed_data["text"]
                                            content += content_delta
                                            yield ChatResponse(
                                                message=ChatMessage(
                                                    role=MessageRole.ASSISTANT,
                                                    content=content,
                                                ),
                                                delta=content_delta,
                                                raw=parsed_data,
                                            )
                        else:
                            raise MaritalkHTTPError(response)

            return gen()

        except ImportError:
            raise ImportError(
                "Could not import httpx python package. "
                "Please install it with `pip install httpx`."
            )

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        astream_complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await astream_complete_fn(prompt, **kwargs)
