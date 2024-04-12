import os
from typing import Any, Callable, Dict, Optional, Sequence, Union
import copy

from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode


class PredibaseLLM(CustomLLM):
    """Predibase LLM.

    To use, you should have the ``predibase`` python package installed,
    and have your Predibase API key.

    The `model_name` parameter is the Predibase "serverless" base_model ID
    (see https://docs.predibase.com/user-guide/inference/models for the catalog).

    An optional `adapter_id` parameter is the HuggingFace ID of a fine-tuned LLM
    adapter, whose base model is the `model` parameter; the fine-tuned adapter
    must be compatible with its base model; otherwise, an error is raised.

    Examples:
        `pip install llama-index-llms-predibase`

        ```python
        import os

        os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

        from llama_index.llms.predibase import PredibaseLLM

        llm = PredibaseLLM(
            model_name="mistral-7b",
            adapter_id="my-repo/my-adapter",  # optional parameter
            temperature=0.3,
            max_new_tokens=512,
        )
        response = llm.complete("Hello World!")
        print(str(response))
        ```
    """

    model_name: str = Field(description="The Predibase base model to use.")
    predibase_api_key: str = Field(description="The Predibase API key to use.")
    adapter_id: str = Field(
        default=None,
        description="The optional HuggingFace ID of a fine-tuned adapter to use.",
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of context tokens available to the LLM.",
        gt=0,
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        predibase_api_key: Optional[str] = None,
        adapter_id: Optional[str] = None,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        temperature: float = DEFAULT_TEMPERATURE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        predibase_api_key = (
            predibase_api_key
            if predibase_api_key
            else os.environ.get("PREDIBASE_API_TOKEN")
        )
        assert predibase_api_key is not None

        super().__init__(
            model_name=model_name,
            adapter_id=adapter_id,
            predibase_api_key=predibase_api_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_window=context_window,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._client = self.initialize_client(predibase_api_key)

    @staticmethod
    def initialize_client(predibase_api_key: str) -> Any:
        try:
            from predibase import PredibaseClient
            from predibase.pql import get_session
            from predibase.pql.api import Session

            session: Session = get_session(
                token=predibase_api_key,
                gateway="https://api.app.predibase.com/v1",
                serving_endpoint="serving.app.predibase.com",
            )
            return PredibaseClient(session=session)
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install predibase`."
            ) from e
        except ValueError as e:
            raise ValueError("Your API key is not correct. Please try again") from e

    @classmethod
    def class_name(cls) -> str:
        return "PredibaseLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> "CompletionResponse":
        from predibase.resource.llm.interface import (
            HuggingFaceLLM,
            LLMDeployment,
        )
        from predibase.resource.llm.response import GeneratedResponse

        base_llm_deployment: LLMDeployment = self._client.LLM(
            uri=f"pb://deployments/{self.model_name}"
        )

        options: Dict[str, Union[str, float]] = copy.deepcopy(kwargs)
        options.update(
            {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
        )

        result: GeneratedResponse
        if self.adapter_id:
            adapter_model: HuggingFaceLLM = self._client.LLM(
                uri=f"hf://{self.adapter_id}"
            )
            result = base_llm_deployment.with_adapter(model=adapter_model).generate(
                prompt=prompt,
                options=options,
            )
        else:
            result = base_llm_deployment.generate(
                prompt=prompt,
                options=options,
            )

        return CompletionResponse(text=result.response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> "CompletionResponseGen":
        raise NotImplementedError
