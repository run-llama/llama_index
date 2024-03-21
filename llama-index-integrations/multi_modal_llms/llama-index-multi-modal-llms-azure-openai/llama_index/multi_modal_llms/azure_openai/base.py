from typing import Any, Callable, Dict, Optional, Tuple

import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata
from llama_index.llms.azure_openai.utils import (
    refresh_openai_azuread_token,
    resolve_from_aliases,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from openai.lib.azure import AsyncAzureOpenAI
from openai.lib.azure import AzureOpenAI as SyncAzureOpenAI


class AzureOpenAIMultiModal(OpenAIMultiModal):
    """
    Azure OpenAI.

    To use this, you must first deploy a model on Azure OpenAI.
    Unlike OpenAI, you need to specify a `engine` parameter to identify
    your deployment (called "model deployment name" in Azure portal).

    - model: Name of the model (e.g. `text-davinci-003`)
        This in only used to decide completion vs. chat endpoint.
    - engine: This will correspond to the custom name you chose
        for your deployment when you deployed a model.

    You must have the following environment variables set:
    - `OPENAI_API_VERSION`: set this to `2023-05-15`
        This may change in the future.
    - `AZURE_OPENAI_ENDPOINT`: your endpoint should look like the following
        https://YOUR_RESOURCE_NAME.openai.azure.com/
    - `AZURE_OPENAI_API_KEY`: your API key if the api type is `azure`

    More information can be found here:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python
    """

    engine: str = Field(description="The name of the deployed azure engine.")
    azure_endpoint: Optional[str] = Field(
        default=None, description="The Azure endpoint to use."
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="The Azure deployment to use."
    )
    use_azure_ad: bool = Field(
        description="Indicates if Microsoft Entra ID (former Azure AD) is used for token authentication"
    )

    _azure_ad_token: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        engine: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: Optional[int] = 300,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        max_retries: int = 3,
        timeout: float = 60.0,
        image_detail: str = "low",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        # azure specific
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        use_azure_ad: bool = False,
        # aliases for engine
        deployment_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        deployment: Optional[str] = None,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        engine = resolve_from_aliases(
            engine, deployment_name, deployment_id, deployment, azure_deployment
        )

        if engine is None:
            raise ValueError("You must specify an `engine` parameter.")

        azure_endpoint = get_from_param_or_env(
            "azure_endpoint", azure_endpoint, "AZURE_OPENAI_ENDPOINT", ""
        )
        super().__init__(
            engine=engine,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_kwargs=additional_kwargs,
            context_window=context_window,
            max_retries=max_retries,
            timeout=timeout,
            image_detail=image_detail,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            use_azure_ad=use_azure_ad,
            default_headers=default_headers,
            http_client=http_client,
            **kwargs,
        )

    def _get_clients(self, **kwargs: Any) -> Tuple[SyncAzureOpenAI, AsyncAzureOpenAI]:
        client = SyncAzureOpenAI(**self._get_credential_kwargs())
        aclient = AsyncAzureOpenAI(**self._get_credential_kwargs())
        return client, aclient

    @classmethod
    def class_name(cls) -> str:
        return "azure_openai_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_new_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.engine,
        )

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.use_azure_ad:
            self._azure_ad_token = refresh_openai_azuread_token(self._azure_ad_token)
            self.api_key = self._azure_ad_token.token

        return {
            "api_key": self.api_key or None,
            "max_retries": self.max_retries,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
            "timeout": self.timeout,
        }

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        model_kwargs["model"] = self.engine
        return model_kwargs
