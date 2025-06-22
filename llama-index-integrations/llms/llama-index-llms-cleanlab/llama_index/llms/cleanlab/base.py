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
from cleanlab_tlm.utils.config import (
    get_default_model,
    get_default_quality_preset,
    get_default_context_limit,
    get_default_max_tokens,
)

DEFAULT_MODEL = get_default_model()
DEFAULT_QUALITY_PRESET = get_default_quality_preset()
DEFAULT_MAX_TOKENS = get_default_max_tokens()


class CleanlabTLM(CustomLLM):
    """
    Cleanlab TLM.

    Examples:
        `pip install llama-index-llms-cleanlab`

        ```python
        from llama_index.llms.cleanlab import CleanlabTLM

        llm = CleanlabTLM(api_key=api_key, quality_preset="best", options={"log": ["explanation"]})
        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```

    Arguments:
    `quality_preset` and `options` are configuration settings you can optionally specify to improve latency or accuracy.

    More information can be found here:
        https://help.cleanlab.ai/tlm/

    """

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Base LLM to use with TLM.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate in TLM response.",
    )
    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        quality_preset: Optional[str] = DEFAULT_QUALITY_PRESET,
        options: Optional[Dict] = None,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
        )

        self.max_tokens = (
            options.get("max_tokens")
            if options and "max_tokens" in options
            else DEFAULT_MAX_TOKENS
        )

        api_key = get_from_param_or_env("api_key", api_key, "CLEANLAB_API_KEY")

        self._client = TLM(
            api_key=api_key, quality_preset=quality_preset, options=options
        )
        self.model = self._client.get_model_name()

    @classmethod
    def class_name(cls) -> str:
        return "CleanlabTLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=get_default_context_limit(),
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
        if "log" in response and "explanation" in response["log"]:
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
