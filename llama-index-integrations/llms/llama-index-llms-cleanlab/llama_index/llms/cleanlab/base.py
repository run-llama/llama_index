from typing import Any, Dict
import json

# Import LlamaIndex dependencies
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.bridge.pydantic import Field

from cleanlab_studio import Studio

class CleanlabTLM(CustomLLM):
    
    # TODO: figure context_window from the underlying model (GPT-3.5 has 16k, GPT-4 has 128k)
    context_window: int = 16000
    max_tokens: int = 512
    model: str = "TLM"

    def __init__(
        self,
        api_key: Optional[str] = None,
        quality_preset: Optional[str] = "medium",
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize params."""
        additional_kwargs = additional_kwargs or {}

        api_key = get_from_param_or_env("api_key", api_key, "CLEANLAB_API_KEY")
        self._api_key = api_key

        super().__init__(
            model=model,
            maxTokens=maxTokens,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Prompt TLM for a response and trustworthiness score
        response: Dict[str, str] = tlm.prompt(prompt)
        output = json.dumps(response)
        return CompletionResponse(text=output)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Prompt TLM for a response and trustworthiness score
        response = tlm.prompt(prompt)
        output = json.dumps(response)

        # Stream the output
        output_str = ""
        for token in output:
            output_str += token
            yield CompletionResponse(text=output_str, delta=token)

class Databricks(OpenAILike):
    """Databricks LLM.

    Examples:
        `pip install llama-index-llms-databricks`

        ```python
        from llama_index.llms.databricks import Databricks

        # Set up the Databricks class with the required model, API key and serving endpoint
        llm = Databricks(model="databricks-dbrx-instruct", api_key="your_api_key", api_base="https://[your-work-space].cloud.databricks.com/serving-endpoints")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of open source LLMs")

        print(response)
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("DATABRICKS_TOKEN", None)
        api_base = api_base or os.environ.get("DATABRICKS_SERVING_ENDPOINT", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Databricks"
