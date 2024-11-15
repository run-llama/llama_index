from typing import Any, Dict, Sequence, List, Literal, Iterable, AsyncIterable

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    LLMMetadata,
    MessageRole,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.base.llms.generic_utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms.custom import CustomLLM
from llama_index.utils.qianfan import (
    Client,
    APIType,
    get_service_list,
    aget_service_list,
)


class ChatMsg(BaseModel):
    """
    Chat request message, which is the message item of the chat request model.
    """

    role: Literal["user", "assistant"]
    """The role that sends the message."""

    content: str
    """The content of the message."""


class ChatRequest(BaseModel):
    """
    Chat request model.
    """

    messages: List[ChatMsg]
    """Chat message list."""

    system: str = ""
    """Prompt."""

    stream: bool = False
    """Indicate whether to respond in stream or not."""


class ChatResp(BaseModel):
    """
    Chat response model.
    """

    result: str


def build_chat_request(
    stream: bool, messages: Sequence[ChatMessage], **kwargs: Any
) -> ChatRequest:
    """
    Construct a ChatRequest.

    :param messages: The chat message list.
    :param stream: Indicate whether to respond in stream or not.
    :return: The ChatResponse object.
    """
    request = ChatRequest(messages=[], stream=stream)
    for message in messages:
        if message.role == MessageRole.USER:
            msg = ChatMsg(role="user", content=message.content)
            request.messages.append(msg)
        elif message.role == MessageRole.ASSISTANT:
            msg = ChatMsg(role="assistant", content=message.content)
            request.messages.append(msg)
        elif message.role == MessageRole.SYSTEM:
            request.system = message.content
        else:
            raise NotImplementedError(
                f"The message role {message.role} is not supported."
            )
    return request


def parse_chat_response(resp_dict: Dict) -> ChatResponse:
    """
    Parse chat response.

    :param resp_dict: Response body in dict form.
    :return: The ChatResponse object.
    """
    resp = ChatResp(**resp_dict)
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=resp.result)
    )


def parse_stream_chat_response(
    resp_dict_iter: Iterable[Dict],
) -> Iterable[ChatResponse]:
    """
    Parse streaming chat response.

    :param resp_dict_iter: Iterator of the response body in dict form.
    :return: Iterator of the ChatResponse object.
    """
    content = ""
    for resp_dict in resp_dict_iter:
        resp = ChatResp(**resp_dict)
        content += resp.result
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            delta=resp.result,
        )


async def aparse_stream_chat_response(
    resp_dict_iter: AsyncIterable[Dict],
) -> AsyncIterable[ChatResponse]:
    """
    Parse asyncio streaming chat response.

    :param resp_dict_iter: Async iterator of the response body in dict form.
    :return: Async iterator of the ChatResponse object.
    """
    content = ""
    async for resp_dict in resp_dict_iter:
        resp = ChatResp(**resp_dict)
        content += resp.result
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            delta=resp.result,
        )


class Qianfan(CustomLLM):
    """
    The LLM supported by Baidu Intelligent Cloud's QIANFAN LLM Platform.
    """

    access_key: str = Field(
        description="The Access Key obtained from the Security Authentication Center of Baidu Intelligent Cloud Console."
    )

    secret_key: str = Field(description="The Secret Key paired with the Access Key.")

    model_name: str = Field(description="The name of the model service.")

    endpoint_url: str = Field(description="The chat endpoint URL of the model service.")

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW, description="The context window size."
    )

    llm_type: APIType = Field(default="chat", description="The LLM type.")

    _client = PrivateAttr()

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        model_name: str,
        endpoint_url: str,
        context_window: int,
        llm_type: APIType = "chat",
    ) -> None:
        """
        Initialize a Qianfan LLM instance.

        :param access_key: The Access Key obtained from the Security Authentication Center
            of Baidu Intelligent Cloud Console.
        :param secret_key: The Secret Key paired with the Access Key.
        :param model_name: The name of the model service. For example: ERNIE-4.0-8K.
        :param endpoint_url: The chat endpoint URL of the model service.
            For example: https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro .
        :param context_windows: The context window size. for example: 8192.
        :param llm_type: The LLM type. Currently, only the chat type is supported.
        """
        if llm_type != "chat":
            raise NotImplementedError("Only the chat type is supported.")

        super().__init__(
            model_name=model_name,
            endpoint_url=endpoint_url,
            context_window=context_window,
            access_key=access_key,
            secret_key=secret_key,
            llm_type=llm_type,
        )
        self._client = Client(access_key, secret_key)

    @classmethod
    def from_model_name(
        cls,
        access_key: str,
        secret_key: str,
        model_name: str,
        context_window: int,
    ):
        """
        Initialize a Qianfan LLM instance. Then query more parameters based on the model name.

        :param access_key: The Access Key obtained from the Security Authentication Center
            of Baidu Intelligent Cloud Console.
        :param secret_key: The Secret Key paired with the Access Key.
        :param model_name: The name of the model service. For example: ERNIE-4.0-8K.
        :param context_windows: The context window size. for example: 8192.
        """
        service_list = get_service_list(access_key, secret_key, ["chat"])
        try:
            service = next(
                service for service in service_list if service.name == model_name
            )
        except StopIteration:
            raise NameError(f"not found {model_name}")

        return cls(
            access_key=access_key,
            secret_key=secret_key,
            model_name=model_name,
            endpoint_url=service.url,
            context_window=context_window,
            llm_type=service.api_type,
        )

    @classmethod
    async def afrom_model_name(
        cls,
        access_key: str,
        secret_key: str,
        model_name: str,
        context_window: int,
    ):
        """
        Initialize a Qianfan LLM instance. Then asynchronously query more parameters based on the model name.

        :param access_key: The Access Key obtained from the Security Authentication Center of
            Baidu Intelligent Cloud Console.
        :param secret_key: The Secret Key paired with the Access Key.
        :param model_name: The name of the model service. For example: ERNIE-4.0-8K.
        :param context_windows: The context window size. for example: 8192.
            The LLMs developed by Baidu all carry context window size in their names.
        """
        service_list = await aget_service_list(access_key, secret_key, ["chat"])
        try:
            service = next(
                service for service in service_list if service.name == model_name
            )
        except StopIteration:
            raise NameError(f"not found {model_name}")

        return cls(
            access_key=access_key,
            secret_key=secret_key,
            model_name=model_name,
            endpoint_url=service.url,
            context_window=context_window,
            llm_type=service.api_type,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Qianfan_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            is_chat_model=self.llm_type == "chat",
            model_name=self.model_name,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Request a chat.

        :param messages: The chat message list. The last message is the current request,
            and the previous messages are the historical chat information. The number of
            members must be odd, and the role value of the odd-numbered messages must be
            "user", while the role value of the even-numbered messages must be "assistant".
        :return: The ChatResponse object.
        """
        request = build_chat_request(stream=False, messages=messages, **kwargs)
        resp_dict = self._client.post(self.endpoint_url, json=request.dict())
        return parse_chat_response(resp_dict)

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Asynchronous request for a chat.

        :param messages: The chat message list. The last message is the current request,
            and the previous messages are the historical chat information. The number of
            members must be odd, and the role value of the odd-numbered messages must be
            "user", while the role value of the even-numbered messages must be "assistant".
        :return: The ChatResponse object.
        """
        request = build_chat_request(stream=False, messages=messages, **kwargs)
        resp_dict = await self._client.apost(self.endpoint_url, json=request.dict())
        return parse_chat_response(resp_dict)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Request a chat, and the response is returned in a stream.

        :param messages: The chat message list. The last message is the current request,
            and the previous messages are the historical chat information. The number of
            members must be odd, and the role value of the odd-numbered messages must be
            "user", while the role value of the even-numbered messages must be "assistant".
        :return: A ChatResponseGen object, which is a generator of ChatResponse.
        """
        request = build_chat_request(stream=True, messages=messages, **kwargs)

        def gen():
            resp_dict_iter = self._client.post_reply_stream(
                self.endpoint_url, json=request.dict()
            )
            yield from parse_stream_chat_response(resp_dict_iter)

        return gen()

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """
        Asynchronous request a chat, and the response is returned in a stream.

        :param messages: The chat message list. The last message is the current request,
            and the previous messages are the historical chat information. The number of
            members must be odd, and the role value of the odd-numbered messages must be
            "user", while the role value of the even-numbered messages must be "assistant".
        :return: A ChatResponseAsyncGen object, which is a asynchronous generator of ChatResponse.
        """
        request = build_chat_request(stream=True, messages=messages, **kwargs)

        async def gen():
            resp_dict_iter = self._client.apost_reply_stream(
                self.endpoint_url, json=request.dict()
            )
            async for part in aparse_stream_chat_response(resp_dict_iter):
                yield part

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Request to complete a message that begins with the specified prompt.
        The LLM developed by Baidu does not support the complete function.
        Here use a converter to convert the chat function to a complete function.

        :param prompt: The prompt message at the beginning of the completed content.
        :return: CompletionResponse.
        """
        complete_fn = chat_to_completion_decorator(self.chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Asynchronous request to complete a message that begins with the specified prompt.
        The LLM developed by Baidu does not support the complete function.
        Here use a converter to convert the chat function to a complete function.

        :param prompt: The prompt message at the beginning of the completed content.
        :return: A CompletionResponse object.
        """
        complete_fn = achat_to_completion_decorator(self.achat)
        return await complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Request to complete a message that begins with the specified prompt,
        and the response is returned in a stream.
        The LLM developed by Baidu does not support the complete function.
        Here use a converter to convert the chat function to a complete function.

        :param prompt: The prompt message at the beginning of the completed content.
        :return: A CompletionResponseGen object.
        """
        complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        return complete_fn(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """
        Asynchronous request to complete a message that begins with the specified prompt,
        and the response is returned in a stream.
        The LLM developed by Baidu does not support the complete function.
        Here use a converter to convert the chat function to a complete function.

        :param prompt: The prompt message at the beginning of the completed content.
        :return: A CompletionResponseAsyncGen object.
        """
        complete_fn = astream_chat_to_completion_decorator(self.astream_chat)
        return await complete_fn(prompt, **kwargs)
