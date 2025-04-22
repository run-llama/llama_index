from typing import Any, Dict, Optional

# Import LlamaIndex dependencies
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
from llama_index.core.llms.callbacks import llm_completion_callback, CallbackManager
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.bridge.pydantic import PrivateAttr, Field

from cleanlab_tlm import TLM

DEFAULT_CONTEXT_WINDOW = 131072
DEFAULT_MAX_TOKENS = 512
DEFAULT_MODEL = "gpt-4o-mini"

MAP_MODEL_CONTEXT_WINDOW = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4o-mini": 131072,
    "gpt-4o": 131072,
    "o1-mini": 131072,
    "o3-mini": 204800,
    "o1": 204800,
    "gpt-4.1": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-nano": 1047576,
    "o3": 204800,
    # Anthropic
    "claude-3-haiku": 204800,
    "claude-3.5-haiku": 204800,
    "claude-3.5-sonnet": 204800,
    "claude-3.5-sonnet-v2": 204800,
    "claude-3.7-sonnet": 204800,
    # Amazon
    "nova-micro": 131072,
    "nova-lite": 307200,
    "nova-pro": 307200,
}


class CleanlabTLM(CustomLLM):
    """
    Cleanlab TLM.

    Examples:
        `pip install llama-index-llms-cleanlab`

        ```python
        from llama_index.llms.cleanlab import CleanlabTLM

        llm = CleanlabTLM(quality_preset="best", api_key=api_key)
        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```
    """

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate in TLM response.",
    )
    model: str = Field(default=DEFAULT_MODEL, description="The base model to use.")
    quality_preset: str = Field(
        default="medium", description="Pre-defined configuration to use for TLM."
    )
    log: dict = Field(
        default_factory=dict, description="Metadata to log from TLM response."
    )
    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        quality_preset: Optional[str] = "medium",
        options: Optional[Dict] = None,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
        )

        self.quality_preset = quality_preset
        self._configure_model(options)

        api_key = get_from_param_or_env("api_key", api_key, "CLEANLAB_API_KEY")

        self._client = TLM(
            api_key=api_key, quality_preset=self.quality_preset, options=options
        )

    def _configure_model(self, options: Optional[Dict]) -> None:
        """Configure model-specific parameters based on provided options."""
        if options and options.get("model"):
            self.model = options["model"]
        else:
            self.model = DEFAULT_MODEL

        self.context_window = MODEL_CONTEXT_WINDOWS.get(
            self.model, DEFAULT_CONTEXT_WINDOW
        )
        self.max_tokens = (
            options.get("max_tokens")
            if options and "max_tokens" in options
            else DEFAULT_MAX_TOKENS
        )

        if options and options.get("log") and "explanation" in options["log"]:
            self.log["explanation"] = True

    @classmethod
    def class_name(cls) -> str:
        return "CleanlabTLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
        )

    def _parse_response(self, response: Dict) -> CompletionResponse:
        """Parse the response from TLM and return a CompletionResponse object."""
        try:
            text = response["response"]
            trust_score = response["trustworthiness_score"]
        except KeyError as e:
            raise ValueError(f"Missing expected key in response: {e}")

        additional_data = {"trustworthiness_score": trust_score}
        if self.log.get("explanation") and "explanation" in response["log"]:
            additional_data["explanation"] = response["log"]["explanation"]

        return CompletionResponse(text=text, additional_kwargs=additional_data)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self._client.prompt(prompt)
        return self._parse_response(response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Raise implementation error since TLM doesn't support native streaming
        raise NotImplementedError(
            "Streaming is not supported in TLM. Instead stream in the response from the LLM and subsequently use TLM to score its trustworthiness."
        )
