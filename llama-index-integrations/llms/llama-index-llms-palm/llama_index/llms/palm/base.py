"""Palm API."""

import os
from typing import Any, Callable, Optional, Sequence

import google.generativeai as palm
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

DEFAULT_PALM_MODEL = "models/text-bison-001"


class PaLM(CustomLLM):
    """
    PaLM LLM.

    Examples:
        `pip install llama-index-llms-palm`

        ```python
        import google.generativeai as palm

        # API key for PaLM
        palm_api_key = "YOUR_API_KEY_HERE"

        # List all models that support text generation
        models = [
            m
            for m in palm.list_models()
            if "generateText" in m.supported_generation_methods
        ]
        model = models[0].name
        print(model)

        # Start using our PaLM LLM abstraction
        from llama_index.llms.palm import PaLM

        # Create an instance of the PaLM class with the API key
        llm = PaLM(model_name=model, api_key=palm_api_key)

        # Use the complete method to generate text based on a prompt
        response = llm.complete("Your prompt text here.")
        print(str(response))
        ```

    """

    model_name: str = Field(
        default=DEFAULT_PALM_MODEL, description="The PaLM model to use."
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = DEFAULT_PALM_MODEL,
        num_output: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        **generate_kwargs: Any,
    ) -> None:
        """Initialize params."""
        api_key = api_key or os.environ.get("PALM_API_KEY")
        palm.configure(api_key=api_key)

        models = palm.list_models()
        models_dict = {m.name: m for m in models}
        if model_name not in models_dict:
            raise ValueError(
                f"Model name {model_name} not found in {models_dict.keys()}"
            )

        model_name = model_name
        model = models_dict[model_name]

        # get num_output
        num_output = num_output or model.output_token_limit

        generate_kwargs = generate_kwargs or {}
        super().__init__(
            model_name=model_name,
            num_output=num_output,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )
        self._model = model

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
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Predict the answer to a query.

        Args:
            prompt (str): Prompt to use for prediction.

        Returns:
            Tuple[str, str]: Tuple of the predicted answer and the formatted prompt.

        """
        completion = palm.generate_text(
            model=self.model_name,
            prompt=prompt,
            **kwargs,
        )
        return CompletionResponse(text=completion.result, raw=completion.candidates[0])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Stream the answer to a query.

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
