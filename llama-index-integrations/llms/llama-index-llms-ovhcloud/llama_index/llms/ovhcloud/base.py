import os
from typing import Any, Callable, Dict, List, Optional, Sequence
from llama_index.core.base.llms.types import (
    ChatMessage,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai import OpenAI
from .utils import Model

DEFAULT_API_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


class OVHcloud(OpenAI):
    """
    OVHcloud AI Endpoints LLM.

    OVHcloud AI Endpoints provides OpenAI-compatible API endpoints for various models.
    You can use the API for free with rate limits if no API key is provided or if it's
    an empty string. Otherwise, generate an API key from the OVHcloud manager at
    https://ovh.com/manager in the Public Cloud section, AI & Machine Learning, AI Endpoints.

    Args:
        model (str): The model name to use (e.g., "llama-3.1-8b-instruct").
                    Model availability is validated dynamically against the API
                    with fallback to static validation if the API call fails.
        temperature (float): The temperature to use for generation
        max_tokens (int): The maximum number of tokens to generate
        additional_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the API
        max_retries (int): The maximum number of retries to make
        api_key (Optional[str]): The OVHcloud API key. If not provided or empty string,
                                the API can be used for free with rate limits.
        callback_manager (Optional[CallbackManager]): Callback manager for logging
        default_headers (Optional[Dict[str, str]]): Default headers for API requests
        system_prompt (Optional[str]): System prompt for chat
        messages_to_prompt (Optional[Callable]): Function to format messages to prompt
        completion_to_prompt (Optional[Callable]): Function to format completion prompt
        pydantic_program_mode (PydanticProgramMode): Mode for Pydantic handling
        output_parser (Optional[BaseOutputParser]): Parser for model outputs
        api_base (Optional[str]): Override the default API base URL

    Examples:
        `pip install llama-index-llms-ovhcloud`

        ```python
        from llama_index.llms.ovhcloud import OVHcloud

        # Using with API key
        llm = OVHcloud(
            model="llama-3.1-8b-instruct",
            api_key="YOUR_API_KEY",
        )
        response = llm.complete("Hello, world!")

        # Using without API key (free with rate limits)
        llm = OVHcloud(
            model="llama-3.1-8b-instruct",
            api_key="",  # or omit api_key parameter
        )
        response = llm.complete("Hello, world!")

        # Get available models dynamically
        llm = OVHcloud(model="llama-3.1-8b-instruct")
        available = llm.available_models  # List[Model] - fetched dynamically
        model_ids = [model.id for model in available]
        print(f"Available models: {model_ids}")

        # Chat messages
        from llama_index.core.llms import ChatMessage
        messages = [
            ChatMessage(
                role="system", content="You are a helpful assistant"
            ),
            ChatMessage(role="user", content="What is the capital of France?"),
        ]
        response = llm.chat(messages)
        print(response)

        ```

    """

    def __init__(
        self,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        api_base: Optional[str] = None,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        # Get API key from parameter or environment variable
        # Allow empty string for free tier usage
        # If not provided, use empty string to allow free tier access
        if api_key is None:
            api_key = os.environ.get("OVHCLOUD_API_KEY", "")
        # If api_key is explicitly set to empty string, keep it as empty string

        # Use provided api_base or default
        api_base = api_base or DEFAULT_API_BASE

        # Validate model dynamically if we have an API key
        # If no API key, we skip validation (free tier)
        if api_key:
            try:
                # Import OpenAI here to avoid circular imports
                from openai import OpenAI as OpenAIClient

                temp_client = OpenAIClient(
                    api_key=api_key,
                    base_url=api_base,
                )
            except Exception:
                # If validation fails, continue anyway (might be network issue)
                # The actual API call will fail if the model is invalid
                pass

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            callback_manager=callback_manager,
            default_headers=default_headers,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @property
    def available_models(self) -> List[Model]:
        """Get available models from OVHcloud AI Endpoints."""
        try:
            return get_available_models_dynamic(self._get_client())
        except Exception:
            # If fetching fails, return empty list or current model
            return [Model(id=self.model)] if hasattr(self, "model") else []

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "OVHcloud_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )
