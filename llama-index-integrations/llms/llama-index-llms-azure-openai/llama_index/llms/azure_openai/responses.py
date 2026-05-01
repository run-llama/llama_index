import os
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.azure_openai.utils import (
    refresh_openai_azuread_token,
    resolve_from_aliases,
)
from llama_index.llms.openai.responses import OpenAIResponses
from openai import AsyncAzureOpenAI
from openai import AzureOpenAI as SyncAzureOpenAI
from openai.lib.azure import AzureADTokenProvider


class AzureOpenAIResponses(OpenAIResponses):
    """
    Azure OpenAI Responses API.

    To use this, you must first deploy a model on Azure OpenAI.
    Unlike OpenAI, you need to specify a `engine` parameter to identify
    your deployment (called "model deployment name" in Azure portal).

    - model: Name of the model (e.g. `gpt-4o`)
    - engine: This will correspond to the custom name you chose
        for your deployment when you deployed a model.

    You must have the following environment variables set:

    - `OPENAI_API_VERSION`: set this to `2025-03-01-preview` or newer.
        This may change in the future.
    - `AZURE_OPENAI_ENDPOINT`: your endpoint should look like the following
        https://YOUR_RESOURCE_NAME.openai.azure.com/
    - `AZURE_OPENAI_API_KEY`: your API key if the api type is `azure`

    More information can be found here:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python

    Examples:
        `pip install llama-index-llms-azure-openai`

        ```python
        from llama_index.llms.azure_openai import AzureOpenAIResponses

        aoai_api_key = "YOUR_AZURE_OPENAI_API_KEY"
        aoai_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"
        aoai_api_version = "2025-03-01-preview"

        llm = AzureOpenAIResponses(
            engine="AZURE_OPENAI_DEPLOYMENT_NAME",
            model="YOUR_AZURE_OPENAI_MODEL_NAME",
            api_key=aoai_api_key,
            azure_endpoint=aoai_endpoint,
            api_version=aoai_api_version,
        )
        ```

    """

    engine: str = Field(description="The name of the deployed azure engine.")
    azure_endpoint: Optional[str] = Field(
        default=None, description="The Azure endpoint to use."
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="The Azure deployment to use."
    )
    use_azure_ad: bool = Field(
        default=False,
        description="Indicates if Microsoft Entra ID (former Azure AD) is used for token authentication",
    )
    azure_ad_token_provider: Optional[AzureADTokenProvider] = Field(
        default=None, description="Callback function to provide Azure Entra ID token."
    )

    _azure_ad_token: Any = PrivateAttr(default=None)
    _client: SyncAzureOpenAI = PrivateAttr()
    _aclient: AsyncAzureOpenAI = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-4o",
        engine: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        reasoning_options: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        track_previous_responses: bool = False,
        store: bool = False,
        built_in_tools: Optional[List[dict]] = None,
        truncation: str = "disabled",
        user: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        call_metadata: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        api_base: Optional[str] = None,
        # azure specific
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_ad_token_provider: Optional[AzureADTokenProvider] = None,
        use_azure_ad: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        # aliases for engine
        deployment_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        deployment: Optional[str] = None,
        # custom httpx client
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        default_headers: Optional[Dict[str, str]] = None,
        context_window: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        engine = resolve_from_aliases(
            engine, deployment_name, deployment_id, deployment, azure_deployment
        )

        if engine is None:
            raise ValueError("You must specify an `engine` parameter.")

        if api_base is None:
            azure_endpoint = get_from_param_or_env(
                "azure_endpoint", azure_endpoint, "AZURE_OPENAI_ENDPOINT", ""
            )

        # Resolve the API key before building clients
        if use_azure_ad:
            if azure_ad_token_provider:
                api_key = azure_ad_token_provider()
            else:
                self._azure_ad_token = refresh_openai_azuread_token(None)
                api_key = self._azure_ad_token.token
        else:
            api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if api_key is None:
            raise ValueError(
                "You must set an `api_key` parameter. "
                "Alternatively, you can set the AZURE_OPENAI_API_KEY env var OR set `use_azure_ad=True`."
            )

        # Build Azure clients before calling super().__init__() to avoid the
        # parent trying to create plain OpenAI clients with Azure-specific kwargs.
        credential_kwargs = {
            "api_key": api_key,
            "max_retries": max_retries,
            "timeout": timeout,
            "azure_endpoint": azure_endpoint,
            "azure_deployment": azure_deployment,
            "base_url": api_base if api_base else None,
            "azure_ad_token_provider": azure_ad_token_provider,
            "api_version": api_version,
            "default_headers": default_headers,
        }

        sync_client = SyncAzureOpenAI(**credential_kwargs, http_client=http_client)
        async_client = AsyncAzureOpenAI(
            **credential_kwargs, http_client=async_http_client
        )

        super().__init__(
            engine=engine,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_options=reasoning_options,
            include=include,
            instructions=instructions,
            track_previous_responses=track_previous_responses,
            store=store,
            built_in_tools=built_in_tools,
            truncation=truncation,
            user=user,
            previous_response_id=previous_response_id,
            call_metadata=call_metadata,
            strict=strict,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_base=api_base,
            azure_ad_token_provider=azure_ad_token_provider,
            use_azure_ad=use_azure_ad,
            api_version=api_version,
            callback_manager=callback_manager,
            http_client=http_client,
            async_http_client=async_http_client,
            default_headers=default_headers,
            context_window=context_window,
            openai_client=sync_client,
            async_openai_client=async_client,
            **kwargs,
        )

        self._http_client = http_client
        self._async_http_client = async_http_client

    def _get_credential_kwargs(
        self, is_async: bool = False, **kwargs: Any
    ) -> Dict[str, Any]:
        if self.use_azure_ad:
            if self.azure_ad_token_provider:
                self.api_key = self.azure_ad_token_provider()
            else:
                self._azure_ad_token = refresh_openai_azuread_token(
                    self._azure_ad_token
                )
                self.api_key = self._azure_ad_token.token
        else:
            self.api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if self.api_key is None:
            raise ValueError(
                "You must set an `api_key` parameter. "
                "Alternatively, you can set the AZURE_OPENAI_API_KEY env var OR set `use_azure_ad=True`."
            )

        return {
            "api_key": self.api_key,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "base_url": self.api_base if self.api_base else None,
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "api_version": self.api_version,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
            **kwargs,
        }

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        model_kwargs["model"] = self.engine
        return model_kwargs

    def _is_azure_client(self) -> bool:
        return True

    @classmethod
    def class_name(cls) -> str:
        return "azure_openai_responses_llm"
