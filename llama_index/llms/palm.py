"""Palm API."""
import os
from typing import Any, Optional

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_completion_callback,
)
from llama_index.llms.custom import CustomLLM


class PaLM(CustomLLM):
    """PaLM LLM."""

    model_name: str = Field(description="The PaLM model to use.")
    num_output: int = Field(description="The number of tokens to generate.")
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "models/text-bison-001",
        num_output: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **generate_kwargs: Any,
    ) -> None:
        """Initialize params."""
        try:
            import google.generativeai as palm
        except ImportError:
            raise ValueError(
                "PaLM is not installed. "
                "Please install it with `pip install google-generativeai`."
            )
        api_key = api_key or os.environ.get("PALM_API_KEY")
        palm.configure(api_key=api_key)

        models = palm.list_models()
        models_dict = {m.name: m for m in models}
        if model_name not in models_dict:
            raise ValueError(
                f"Model name {model_name} not found in {models_dict.keys()}"
            )

        model_name = model_name
        self._model = models_dict[model_name]

        # get num_output
        num_output = num_output or self._model.output_token_limit

        generate_kwargs = generate_kwargs or {}
        super().__init__(
            model_name=model_name,
            num_output=num_output,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PaLM_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # TODO: google palm actually separates input and output token limits
        total_tokens = self._model.input_token_limit + self.num_output
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Predict the answer to a query.

        Args:
            prompt (str): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        import google.generativeai as palm

        completion = palm.generate_text(
            model=self.model_name,
            prompt=prompt,
            **kwargs,
        )
        return CompletionResponse(text=completion.result, raw=completion.candidates[0])

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (str): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """
        raise NotImplementedError(
            "PaLM does not support streaming completion in LlamaIndex currently."
        )
