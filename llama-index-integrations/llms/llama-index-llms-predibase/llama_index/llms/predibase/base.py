import os
from typing import Any, Callable, Dict, Optional, Sequence, Union

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
    """
    Predibase LLM.

    To use, you should have the ``predibase`` python package installed,
    and have your Predibase API key.

    The `model_name` parameter is the Predibase "serverless" base_model ID
    (see https://docs.predibase.com/user-guide/inference/models for the catalog).

    An optional `adapter_id` parameter is the Predibase ID or the HuggingFace ID
    of a fine-tuned LLM adapter, whose base model is the `model` parameter; the
    fine-tuned adapter must be compatible with its base model; otherwise, an
    error is raised.  If the fine-tuned adapter is hosted at Predibase,
    `adapter_version` must be specified.

    Examples:
        `pip install llama-index-llms-predibase`

        ```python
        import os

        os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"

        from llama_index.llms.predibase import PredibaseLLM

        llm = PredibaseLLM(
            model_name="mistral-7b",
            predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
            adapter_id="my-adapter-id",  # optional parameter
            adapter_version=3,  # optional parameter (applies to Predibase only)
            api_token,  # optional parameter for accessing services hosting adapters (e.g., HuggingFace)
            temperature=0.3,
            max_new_tokens=512,
        )
        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

    model_name: str = Field(description="The Predibase base model to use.")
    predibase_api_key: str = Field(description="The Predibase API key to use.")
    predibase_sdk_version: str = Field(
        default=None,
        description="The optional version (string) of the Predibase SDK (defaults to the latest if not specified).",
    )
    adapter_id: str = Field(
        default=None,
        description="The optional Predibase ID or HuggingFace ID of a fine-tuned adapter to use.",
    )
    adapter_version: str = Field(
        default=None,
        description="The optional version number of fine-tuned adapter use (applies to Predibase only).",
    )
    api_token: str = Field(
        default=None,
        description="The adapter hosting service API key to use.",
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
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
        predibase_sdk_version: Optional[str] = None,
        adapter_id: Optional[str] = None,
        adapter_version: Optional[int] = None,
        api_token: Optional[str] = None,
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
        if not predibase_api_key:
            raise ValueError(
                'Your "PREDIBASE_API_TOKEN" is empty.  Please generate a valid "PREDIBASE_API_TOKEN" in your Predibase account.'
            )

        super().__init__(
            model_name=model_name,
            predibase_api_key=predibase_api_key,
            predibase_sdk_version=predibase_sdk_version,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            api_token=api_token,
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

        self._client: Union["PredibaseClient", "Predibase"] = self.initialize_client()

    def initialize_client(
        self,
    ) -> Union["PredibaseClient", "Predibase"]:
        try:
            if self._is_deprecated_sdk_version():
                from predibase import PredibaseClient
                from predibase.pql import get_session
                from predibase.pql.api import Session

                session: Session = get_session(
                    token=self.predibase_api_key,
                    gateway="https://api.app.predibase.com/v1",
                    serving_endpoint="serving.app.predibase.com",
                )
                return PredibaseClient(session=session)

            from predibase import Predibase

            os.environ["PREDIBASE_GATEWAY"] = "https://api.app.predibase.com"
            return Predibase(api_token=self.predibase_api_key)
        except ValueError as e:
            raise ValueError(
                'Your "PREDIBASE_API_TOKEN" is not correct.  Please try again.'
            ) from e

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
        options: Dict[str, Union[str, float]] = {
            **{
                "api_token": self.api_token,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
            **(kwargs or {}),
        }

        response_text: str

        if self._is_deprecated_sdk_version():
            from predibase.pql.api import ServerResponseError
            from predibase.resource.llm.interface import (
                HuggingFaceLLM,
                LLMDeployment,
            )
            from predibase.resource.llm.response import GeneratedResponse
            from predibase.resource.model import Model

            base_llm_deployment: LLMDeployment = self._client.LLM(
                uri=f"pb://deployments/{self.model_name}"
            )

            result: GeneratedResponse
            if self.adapter_id:
                """
                Attempt to retrieve the fine-tuned adapter from a Predibase repository.
                If absent, then load the fine-tuned adapter from a HuggingFace repository.
                """
                adapter_model: Union[Model, HuggingFaceLLM]
                try:
                    adapter_model = self._client.get_model(
                        name=self.adapter_id,
                        version=self.adapter_version,
                        model_id=None,
                    )
                except ServerResponseError:
                    # Predibase does not recognize the adapter ID (query HuggingFace).
                    adapter_model = self._client.LLM(uri=f"hf://{self.adapter_id}")
                result = base_llm_deployment.with_adapter(model=adapter_model).generate(
                    prompt=prompt,
                    options=options,
                )
            else:
                result = base_llm_deployment.generate(
                    prompt=prompt,
                    options=options,
                )
            response_text = result.response
        else:
            import requests
            from lorax.client import Client as LoraxClient
            from lorax.errors import GenerationError
            from lorax.types import Response

            lorax_client: LoraxClient = self._client.deployments.client(
                deployment_ref=self.model_name
            )

            response: Response
            if self.adapter_id:
                """
                Attempt to retrieve the fine-tuned adapter from a Predibase repository.
                If absent, then load the fine-tuned adapter from a HuggingFace repository.
                """
                if self.adapter_version:
                    # Since the adapter version is provided, query the Predibase repository.
                    pb_adapter_id: str = f"{self.adapter_id}/{self.adapter_version}"
                    options.pop("api_token", None)
                    try:
                        response = lorax_client.generate(
                            prompt=prompt,
                            adapter_id=pb_adapter_id,
                            **options,
                        )
                    except GenerationError as ge:
                        raise ValueError(
                            f'An adapter with the ID "{pb_adapter_id}" cannot be found in the Predibase repository of fine-tuned adapters.'
                        ) from ge
                else:
                    # The adapter version is omitted, hence look for the adapter ID in the HuggingFace repository.
                    try:
                        response = lorax_client.generate(
                            prompt=prompt,
                            adapter_id=self.adapter_id,
                            adapter_source="hub",
                            **options,
                        )
                    except GenerationError as ge:
                        raise ValueError(
                            f"""Either an adapter with the ID "{self.adapter_id}" cannot be found in a HuggingFace repository, \
or it is incompatible with the base model (please make sure that the adapter configuration is consistent).
"""
                        ) from ge
            else:
                try:
                    response = lorax_client.generate(
                        prompt=prompt,
                        **options,
                    )
                except requests.JSONDecodeError as jde:
                    raise ValueError(
                        f"""An LLM with the deployment ID "{self.model_name}" cannot be found at Predibase \
(please refer to "https://docs.predibase.com/user-guide/inference/models" for the list of supported models).
"""
                    ) from jde
            response_text = response.generated_text

        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> "CompletionResponseGen":
        raise NotImplementedError

    def _is_deprecated_sdk_version(self) -> bool:
        try:
            import semantic_version
            from semantic_version.base import Version

            from predibase.version import __version__ as current_version

            sdk_semver_deprecated: Version = semantic_version.Version(
                version_string="2024.4.8"
            )
            actual_current_version: str = self.predibase_sdk_version or current_version
            sdk_semver_current: Version = semantic_version.Version(
                version_string=actual_current_version
            )
            return not (
                (sdk_semver_current > sdk_semver_deprecated)
                or ("+dev" in actual_current_version)
            )
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install semantic_version predibase`."
            ) from e
