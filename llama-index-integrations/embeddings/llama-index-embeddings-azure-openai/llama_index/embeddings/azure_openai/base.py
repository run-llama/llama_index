from typing import Any, Dict, Optional

import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr, root_validator
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingMode,
    OpenAIEmbeddingModelType,
)
from llama_index.llms.azure_openai.utils import resolve_from_aliases
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.lib.azure import AzureADTokenProvider


class AzureOpenAIEmbedding(OpenAIEmbedding):
    azure_endpoint: Optional[str] = Field(
        default=None, description="The Azure endpoint to use."
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="The Azure deployment to use."
    )

    api_base: str = Field(default="", description="The base URL for Azure deployment.")
    api_version: str = Field(
        default="", description="The version for Azure OpenAI API."
    )

    azure_ad_token_provider: AzureADTokenProvider = Field(
        default=None, description="Callback function to provide Azure AD token."
    )

    _client: AzureOpenAI = PrivateAttr()
    _aclient: AsyncAzureOpenAI = PrivateAttr()

    def __init__(
        self,
        mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
        model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        # azure specific
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
        deployment_name: Optional[str] = None,
        max_retries: int = 10,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        num_workers: Optional[int] = None,
        # custom httpx client
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ):
        azure_endpoint = get_from_param_or_env(
            "azure_endpoint", azure_endpoint, "AZURE_OPENAI_ENDPOINT", ""
        )

        api_key = get_from_param_or_env("api_key", api_key, "AZURE_OPENAI_API_KEY")

        azure_deployment = resolve_from_aliases(
            azure_deployment,
            deployment_name,
        )

        super().__init__(
            mode=mode,
            model=model,
            embed_batch_size=embed_batch_size,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            azure_ad_token_provider=azure_ad_token_provider,
            max_retries=max_retries,
            reuse_client=reuse_client,
            callback_manager=callback_manager,
            http_client=http_client,
            num_workers=num_workers,
            **kwargs,
        )

    @root_validator(pre=True)
    def validate_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate necessary credentials are set."""
        if (
            values["api_base"] == "https://api.openai.com/v1"
            and values["azure_endpoint"] is None
        ):
            raise ValueError(
                "You must set OPENAI_API_BASE to your Azure endpoint. "
                "It should look like https://YOUR_RESOURCE_NAME.openai.azure.com/"
            )
        if values["api_version"] is None:
            raise ValueError("You must set OPENAI_API_VERSION for Azure OpenAI.")

        return values

    def _get_client(self) -> AzureOpenAI:
        if not self.reuse_client:
            return AzureOpenAI(**self._get_credential_kwargs())

        if self._client is None:
            self._client = AzureOpenAI(**self._get_credential_kwargs())
        return self._client

    def _get_aclient(self) -> AsyncAzureOpenAI:
        if not self.reuse_client:
            return AsyncAzureOpenAI(**self._get_credential_kwargs())

        if self._aclient is None:
            self._aclient = AsyncAzureOpenAI(**self._get_credential_kwargs())
        return self._aclient

    def _get_credential_kwargs(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
        }

    @classmethod
    def class_name(cls) -> str:
        return "AzureOpenAIEmbedding"
