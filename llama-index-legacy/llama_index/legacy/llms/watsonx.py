from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.generic_utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
)
from llama_index.llms.llm import LLM
from llama_index.llms.watsonx_utils import (
    WATSONX_MODELS,
    get_from_param_or_env_without_error,
    watsonx_model_to_context_size,
)
from llama_index.types import BaseOutputParser, PydanticProgramMode


class WatsonX(LLM):
    """IBM WatsonX LLM."""

    model_id: str = Field(description="The Model to use.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    temperature: float = Field(description="The temperature to use for sampling.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional Kwargs for the WatsonX model"
    )
    model_info: Dict[str, Any] = Field(
        default_factory=dict, description="Details about the selected model"
    )

    _model = PrivateAttr()

    def __init__(
        self,
        credentials: Dict[str, Any],
        model_id: Optional[str] = "ibm/mpt-7b-instruct2",
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        max_new_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """Initialize params."""
        if model_id not in WATSONX_MODELS:
            raise ValueError(
                f"Model name {model_id} not found in {WATSONX_MODELS.keys()}"
            )

        try:
            from ibm_watson_machine_learning.foundation_models.model import Model
        except ImportError as e:
            raise ImportError(
                "You must install the `ibm_watson_machine_learning` package to use WatsonX"
                "please `pip install ibm_watson_machine_learning`"
            ) from e

        additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])

        project_id = get_from_param_or_env_without_error(
            project_id, "IBM_WATSONX_PROJECT_ID"
        )
        space_id = get_from_param_or_env_without_error(space_id, "IBM_WATSONX_SPACE_ID")

        if project_id is not None or space_id is not None:
            self._model = Model(
                model_id=model_id,
                credentials=credentials,
                project_id=project_id,
                space_id=space_id,
            )
        else:
            raise ValueError(
                f"Did not find `project_id` or `space_id`, Please pass them as named parameters"
                f" or as environment variables, `IBM_WATSONX_PROJECT_ID` or `IBM_WATSONX_SPACE_ID`."
            )

        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_kwargs=additional_kwargs,
            model_info=self._model.get_details(),
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(self) -> str:
        """Get Class Name."""
        return "WatsonX_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=watsonx_model_to_context_size(self.model_id),
            num_output=self.max_new_tokens,
            model_name=self.model_id,
        )

    @property
    def sample_model_kwargs(self) -> Dict[str, Any]:
        """Get a sample of Model kwargs that a user can pass to the model."""
        try:
            from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames
        except ImportError as e:
            raise ImportError(
                "You must install the `ibm_watson_machine_learning` package to use WatsonX"
                "please `pip install ibm_watson_machine_learning`"
            ) from e

        params = GenTextParamsMetaNames().get_example_values()

        params.pop("return_options")

        return params

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        return {**base_kwargs, **self.additional_kwargs}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {**self._model_kwargs, **kwargs}

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)

        response = self._model.generate_text(prompt=prompt, params=all_kwargs)

        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)

        stream_response = self._model.generate_text_stream(
            prompt=prompt, params=all_kwargs
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for stream_delta in stream_response:
                content += stream_delta
                yield CompletionResponse(text=content, delta=stream_delta)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_fn = completion_to_chat_decorator(self.complete)

        return chat_fn(messages, **all_kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        all_kwargs = self._get_all_kwargs(**kwargs)
        chat_stream_fn = stream_completion_to_chat_decorator(self.stream_complete)

        return chat_stream_fn(messages, **all_kwargs)

    # Async Functions
    # IBM Watson Machine Learning Package currently does not have Support for Async calls

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
