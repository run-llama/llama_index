import json

import aiohttp
import requests

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    AsyncGenerator,
    Generator,
    Union,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM

from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
    MessageRole,
    ChatMessage,
)

"""DeepInfra API base URL."""
API_BASE = "https://api.deepinfra.com/v1"
"""DeepInfra Inference API endpoint."""
INFERENCE_ENDPOINT = "inference"
"""Chat API endpoint for DeepInfra."""
CHAT_API_ENDPOINT = "openai/chat/completions"
"""Environment variable name of DeepInfra API token."""
ENV_VARIABLE = "DEEPINFRA_API_TOKEN"
"""Default model name for DeepInfra embeddings."""
DEFAULT_MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"


class DeepInfraLLM(LLM):
    """DeepInfra LLM.

    Examples:
        `pip install llama-index-llms-deepinfra`

        ```python
        from llama_index.llms.deepinfra import DeepInfraLLM

        llm = DeepInfraLLM(
            model_name="mistralai/Mixtral-8x22B-Instruct-v0.1", # Default model name
            api_key = "your-deepinfra-api-key",
            temperature=0.5,
            max_tokens=50,
            additional_kwargs={"top_p": 0.9},
        )

        response = llm.complete("Hello World!")
        print(response)
        ```
    """

    model_name: str = Field(
        default=DEFAULT_MODEL_NAME, description="The DeepInfra model to use."
    )

    _api_key: Optional[str] = PrivateAttr()

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )

    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gte=1,
    )

    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments for generation."
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        self._api_key = get_from_param_or_env("api_key", api_key, ENV_VARIABLE)

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=API_BASE,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "DeepInfra_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens,
            is_chat_model=self._is_chat_model,
            model_name=self.model,
        )

    @property
    def _is_chat_model(self) -> bool:
        return True

    # Synchronous Methods
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            str: The generated text completion.
        """
        result = self._request(self.get_model_endpoint(), {"input": prompt, **kwargs})
        return CompletionResponse(
            text=result["results"][0]["generated_text"], raw=result
        )

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """
        Generate a synchronous streaming completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            CompletionResponseGen: The streaming text completion.
        """
        payload = {"model": self.model_name, "input": prompt, **kwargs}

        content = ""
        for response_dict in self._request_stream(self.get_model_endpoint(), payload):
            content_delta = response_dict["token"]["text"]
            content += content_delta
            yield CompletionResponse(
                text=content, delta=content_delta, raw=response_dict
            )

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """
        Generate a chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            ChatResponse: The chat response containing a sequence of messages.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            **kwargs,
        }
        result = self._request(CHAT_API_ENDPOINT, payload)

        return ChatResponse(
            message=ChatMessage(
                role=result["choices"][-1]["message"]["role"],
                content=result["choices"][-1]["message"]["content"],
            ),
            raw=result,
        )

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponseGen:
        """
        Generate a synchronous streaming chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            ChatResponseGen: The chat response containing a sequence of messages.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            **kwargs,
        }

        content = ""
        role = MessageRole.ASSISTANT
        for response_dict in self._request_stream(CHAT_API_ENDPOINT, payload):
            delta = response_dict["choices"][-1]["delta"]
            """
            Check if the delta contains content.
            """
            if delta.get("content", None):
                content_delta = delta["content"]
                content += delta["content"]
                message = ChatMessage(
                    role=role,
                    content=content,
                    delta=content_delta,
                )
                yield ChatResponse(message=message, raw=response_dict)

    # Asynchronous Methods
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Asynchronously generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            CompletionResponse: The generated text completion.
        """
        result = await self._arequest(
            self.get_model_endpoint(), {"input": prompt, **kwargs}
        )
        return CompletionResponse(
            text=result["results"][0]["generated_text"], raw=result
        )

    async def astream_complete(
        self, prompt: str, **kwargs
    ) -> CompletionResponseAsyncGen:
        """
        Asynchronously generate a streaming completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            CompletionResponseAsyncGen: The streaming text completion.
        """
        payload = {"model": self.model_name, "input": prompt, **kwargs}

        content = ""
        async for response_dict in self._arequest_stream(
            self.get_model_endpoint(), payload
        ):
            content_delta = response_dict["token"]["text"]
            content += content_delta
            yield CompletionResponse(
                text=content, delta=content_delta, raw=response_dict
            )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """
        Asynchronously generate a chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            ChatResponse: The chat response containing a sequence of messages.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            **kwargs,
        }
        result = await self._arequest(CHAT_API_ENDPOINT, payload)
        return ChatResponse(
            message=ChatMessage(
                role=result["choices"][-1]["message"]["role"],
                content=result["choices"][-1]["message"]["content"],
            ),
            raw=result,
        )

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponseAsyncGen:
        """
        Asynchronously generate a streaming chat response for the given messages.

        Args:
            messages (Sequence[ChatMessage]): A sequence of chat messages.
            **kwargs: Additional keyword arguments for the API request.

        Yields:
            ChatResponseAsyncGen: The chat response containing a sequence of messages.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            **kwargs,
        }

        content = ""
        role = MessageRole.ASSISTANT
        async for response_dict in self._arequest_stream(CHAT_API_ENDPOINT, payload):
            delta = response_dict["choices"][-1]["delta"]
            """
            Check if the delta contains content.
            """
            if delta.get("content", None):
                content_delta = delta["content"]
                content += delta["content"]
                message = ChatMessage(
                    role=role,
                    content=content,
                    delta=content_delta,
                )
                yield ChatResponse(message=message, raw=response_dict)

    # Private Methods
    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Private method to perform a synchronous request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.get_url(endpoint), json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def _request_stream(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Private method to perform a synchronous streaming request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Yields:
            str: The streaming response from the API.
        """
        payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.get_url(endpoint), json=payload, headers=headers, stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if resp := self.decode_data(line):
                yield resp

    async def _arequest(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Private method to perform an asynchronous request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Returns:
            Dict[str, Any]: The API response.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.get_url(endpoint), json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _arequest_stream(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Private method to perform an asynchronous streaming request to the DeepInfra API.

        Args:
            endpoint (str): The API endpoint to send the request to.
            payload (Dict[str, Any]): The request payload.

        Yields:
            str: The streaming response from the API.
        """
        payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.get_url(endpoint), json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if resp := self.decode_data(line):
                        yield resp

    # Utility Method
    def get_model_endpoint(self) -> str:
        """
        Get DeepInfra model endpoint.
        """
        return f"{INFERENCE_ENDPOINT}/{self.model_name}"

    def get_url(self, endpoint: str) -> str:
        """
        Get DeepInfra API URL.
        """
        return f"{API_BASE}/{endpoint}"

    def decode_data(self, data: bytes) -> Union[dict, None]:
        """
        Decode data from the streaming response.
        Checks whether the incoming data is an actual
        SSE data message.

        Args:
            data (bytes): The incoming data.

        Returns:
            Union[dict, None]: The decoded data or None.
        """
        if data and data.startswith(b"data: "):
            data = data.decode("utf-8").strip("data: ")
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        else:
            return None
