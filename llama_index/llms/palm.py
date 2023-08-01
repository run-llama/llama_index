"""Palm API."""

from llama_index.llms.custom import CustomLLM
from typing import Optional, Any
from llama_index.llms.base import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    llm_callback,
)
import os


class PaLM(CustomLLM):
    """PaLM LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = "models/text-bison-001",
        num_output: Optional[int] = None,
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
        self._api_key = api_key
        palm.configure(api_key=api_key)
        models = palm.list_models()
        models_dict = {m.name: m for m in models}
        if model_name not in models_dict:
            raise ValueError(
                f"Model name {model_name} not found in {models_dict.keys()}"
            )
        self._model_name = model_name
        self._model = models_dict[model_name]

        # get num_output
        self._num_output = num_output or self._model.output_token_limit

        self._generate_kwargs = generate_kwargs

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # TODO: google palm actually separates input and output token limits
        total_tokens = self._model.input_token_limit + self._num_output
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self._num_output,
            model_name=self._model_name,
        )

    @llm_callback(is_chat=False)
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Predict the answer to a query.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """

        import google.generativeai as palm

        completion = palm.generate_text(
            model=self._model_name,
            prompt=prompt,
            **kwargs,
        )
        return CompletionResponse(text=completion.result, raw=completion.candidates[0])

    @llm_callback(is_chat=False)
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream the answer to a query.

        NOTE: this is a beta feature. Will try to build or use
        better abstractions about response handling.

        Args:
            prompt (Prompt): Prompt to use for prediction.

        Returns:
            str: The predicted answer.

        """
        raise NotImplementedError(
            "PaLM does not support streaming completion in LlamaIndex currently."
        )
