import aiohttp
import requests

from typing import Any, Callable, Dict, Optional, Sequence
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM

from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.base.llms.types import ChatMessage, LLMMetadata
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.base.llms.types import (
    CompletionResponseGen,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponseAsyncGen,
    LLMMetadata,
)


"""DeepInfra Inference API URL."""
INFERENCE_URL = "https://api.deepinfra.com/v1/inference"
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
        default_factory=dict,
        description="Additional keyword arguments for generation.",
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
            api_base=INFERENCE_URL,
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
            is_chat_model=True,
            model_name=self.model,
        )

    @property
    def _is_chat_model(self) -> bool:
        return True

    async def acomplete(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            str: The generated text completion.
        """
        result = await self._arequest(self.model_name, {"input": prompt, **kwargs})
        return result["results"][0]["generated_text"]

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            str: The generated text completion.
        """
        result = self._request(self.model_name, {"input": prompt, **kwargs})
        return result["results"][0]["generated_text"]

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
        result = self._request("openai/chat/completions", payload)
        return ChatResponse(
            messages=[
                ChatMessage(
                    role=choice["message"]["role"], content=choice["message"]["content"]
                )
                for choice in result["choices"]
            ]
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
        result = await self._arequest("openai/chat/completions", payload)
        return ChatResponse(
            messages=[
                ChatMessage(
                    role=choice["message"]["role"], content=choice["message"]["content"]
                )
                for choice in result["choices"]
            ]
        )

    def get_url(self):
        """
        Get DeepInfra API URL.
        """
        return f"{INFERENCE_URL}/{self.model_name}"

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
        response = requests.post(
            f"{INFERENCE_URL}/{endpoint}", json=payload, headers=headers
        )
        response.raise_for_status()
        return response.json()

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
                f"{INFERENCE_URL}/{endpoint}", json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def astream_complete(self, question: str) -> CompletionResponseAsyncGen:
        raise NotImplementedError("DeepInfra does not currently support streaming.")

    async def astream_chat(self, question: str) -> ChatResponseAsyncGen:
        raise NotImplementedError("DeepInfra does not currently support streaming.")

    def stream_chat(self, question: str) -> ChatResponseGen:
        raise NotImplementedError("DeepInfra does not currently support chat.")

    def stream_complete(self, question: str) -> CompletionResponseGen:
        raise NotImplementedError(
            "DeepInfra does not currently support streaming completion."
        )
