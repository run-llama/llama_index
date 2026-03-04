from typing import Any, Callable, Dict, List, Optional, Sequence
import aiohttp
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.openai import OpenAI
from llama_index.core.bridge.pydantic import Field
from .utils import validate_model_dynamic, get_available_models_dynamic, Model

DEFAULT_SYNC_API_BASE = (
    "https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)
DEFAULT_ASYNC_API_BASE = (
    "https://model-{model_id}.api.baseten.co/production/async_predict"
)
MODEL_APIS_BASE = "https://inference.baseten.co/v1/"


class Baseten(OpenAI):
    """
    Baseten LLM with support for both dedicated and model apis endpoints.

    Args:
        model_id (str): The Baseten model ID (e.g., "12a3b4c5") or model name (e.g., "deepseek-ai/DeepSeek-V3-0324").
                       When using model_apis=True, model availability is validated dynamically against the API
                       with fallback to static validation if the API call fails.
        model_apis (bool): If True (default), uses the model apis endpoint. If False, uses the dedicated endpoint.
        webhook_endpoint (Optional[str]): Webhook endpoint for async operations. If provided, uses async API.
        temperature (float): The temperature to use for generation
        max_tokens (int): The maximum number of tokens to generate
        additional_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the API
        max_retries (int): The maximum number of retries to make
        api_key (Optional[str]): The Baseten API key
        callback_manager (Optional[CallbackManager]): Callback manager for logging
        default_headers (Optional[Dict[str, str]]): Default headers for API requests
        system_prompt (Optional[str]): System prompt for chat
        messages_to_prompt (Optional[Callable]): Function to format messages to prompt
        completion_to_prompt (Optional[Callable]): Function to format completion prompt
        pydantic_program_mode (PydanticProgramMode): Mode for Pydantic handling
        output_parser (Optional[BaseOutputParser]): Parser for model outputs

    Examples:
        `pip install llama-index-llms-baseten`

        ```python
        from llama_index.llms.baseten import Baseten

        # Using model apis endpoint (default behavior)
        llm = Baseten(
            model_id="deepseek-ai/DeepSeek-V3-0324",
            api_key="YOUR_API_KEY",
            model_apis=True,  # Default
        )
        response = llm.complete("Hello, world!")

        # Using dedicated endpoint (for custom deployed models)
        llm = Baseten(
            model_id="YOUR_MODEL_ID",
            api_key="YOUR_API_KEY",
            model_apis=False,
        )
        response = llm.complete("Hello, world!")

        # Asynchronous usage with webhook (dedicated endpoint only)
        async_llm = Baseten(
            model_id="YOUR_MODEL_ID",
            api_key="YOUR_API_KEY",
            model_apis=False,  # Required for async operations
            webhook_endpoint="https://your-webhook.com/baseten-callback"
        )
        response = await async_llm.acomplete("Hello, world!")
        request_id = response.text  # Track this ID for webhook response

        # Get available models dynamically (Model APIs only)
        llm = Baseten(model_id="deepseek-ai/DeepSeek-V3-0324", model_apis=True)
        available = llm.available_models  # List[Model] - fetched dynamically
        model_ids = [model.id for model in available]
        print(f"Available models: {model_ids}")

        ```

    """

    webhook_endpoint: Optional[str] = Field(
        default=None, description="Webhook endpoint for async operations"
    )
    model_apis: bool = Field(
        default=True,
        description="Whether to use the model apis endpoint or the dedicated endpoint",
    )

    def __init__(
        self,
        model_id: str,
        model_apis: bool = True,
        webhook_endpoint: Optional[str] = None,
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
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        # Validate model_id if using model apis endpoint
        if model_apis:
            # Use dynamic validation with fallback to static validation
            # We need to create a temporary client for validation
            api_key_temp = get_from_param_or_env("api_key", api_key, "BASETEN_API_KEY")

            # Import OpenAI here to avoid circular imports
            from openai import OpenAI as OpenAIClient

            temp_client = OpenAIClient(
                api_key=api_key_temp,
                base_url=MODEL_APIS_BASE,
            )
            validate_model_dynamic(temp_client, model_id)

        # Determine API base URL based on endpoint type
        if model_apis:
            api_base = MODEL_APIS_BASE
        else:
            api_base = DEFAULT_SYNC_API_BASE.format(model_id=model_id)

        api_key = get_from_param_or_env("api_key", api_key, "BASETEN_API_KEY")

        super().__init__(
            model=model_id,  # model_id is either the Baseten model ID or the specific model APIs slug, stored in OpenAI class
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
        # Set webhook endpoint after parent initialization to avoid errors
        self.webhook_endpoint = webhook_endpoint
        self.model_apis = model_apis

    @property
    def available_models(self) -> List[Model]:
        """Get available models from Baseten Model APIs."""
        if not self.model_apis:
            # For dedicated deployments, return current model or empty list
            return [Model(id=self.model)] if hasattr(self, "model") else []

        # For Model APIs, fetch from the API dynamically
        return get_available_models_dynamic(self._get_client())

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Baseten_LLM"

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion - requires webhook_endpoint for async API."""
        if not self.webhook_endpoint:
            raise ValueError(
                "webhook_endpoint must be provided for async operations with Baseten"
            )

        if self.model_apis:
            raise ValueError(
                "Async operations are not supported with model apis endpoints"
            )

        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Api-Key {self.api_key}"}
            payload = {
                "model_input": {
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    **kwargs,
                },
                "webhook_endpoint": self.webhook_endpoint,
            }

            async with session.post(
                DEFAULT_ASYNC_API_BASE.format(model_id=self.model),
                headers=headers,
                json=payload,
            ) as response:
                if response.status not in [200, 201]:
                    raise Exception(
                        f"Error from Baseten API: {await response.text()}, Response status: {response.status}"
                    )

                result = await response.json()
                request_id = result.get("request_id")

                return CompletionResponse(
                    text=request_id,  # Return request_id for tracking
                    raw=result,
                    additional_kwargs={"async_request": True},
                )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,  # Use chat completions for model APIs
        )
