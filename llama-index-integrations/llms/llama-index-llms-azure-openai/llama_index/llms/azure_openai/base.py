from typing import Any, Callable, Dict, Optional, Sequence

import httpx
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr, model_validator
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.llms.azure_openai.utils import (
    refresh_openai_azuread_token,
    resolve_from_aliases,
)
from llama_index.llms.openai import OpenAI
from openai import AsyncAzureOpenAI
from openai import AzureOpenAI as SyncAzureOpenAI
from openai.lib.azure import AzureADTokenProvider


class AzureOpenAI(OpenAI):
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

    - `OPENAI_API_VERSION`: set this to `2023-07-01-preview` or newer.
        This may change in the future.
    - `AZURE_OPENAI_ENDPOINT`: your endpoint should look like the following
        https://YOUR_RESOURCE_NAME.openai.azure.com/
    - `AZURE_OPENAI_API_KEY`: your API key if the api type is `azure`| Or pass through `AZURE_AD_TOKEN_PROVIDER`
        and set `use_azure_ad = True` to use managed identity with Azure Entra ID

    More information can be found here:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python

    Examples:
        `pip install llama-index-llms-azure-openai`

        ```python
        from llama_index.llms.azure_openai import AzureOpenAI

        aoai_api_key = "YOUR_AZURE_OPENAI_API_KEY"
        aoai_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"
        aoai_api_version = "2023-07-01-preview"

        llm = AzureOpenAI(
            engine="AZURE_AZURE_OPENAI_DEPLOYMENT_NAME",
            model="YOUR_AZURE_OPENAI_COMPLETION_MODEL_NAME",
            api_key=aoai_api_key,
            azure_endpoint=aoai_endpoint,
            api_version=aoai_api_version,
        )
        ```

        Using managed identity (passing a token provider instead of an API key):

         ```python
        from llama_index.llms.azure_openai import AzureOpenAI

        aoai_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"
        aoai_api_version = "2023-07-01-preview"

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

        llm = AzureOpenAI(
            engine = llm_deployment
            model="YOUR_AZURE_OPENAI_COMPLETION_MODEL_NAME",
            azure_ad_token_provider=token_provider,
            use_azure_ad=True,
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
        description="Indicates if Microsoft Entra ID (former Azure AD) is used for token authentication"
    )

    azure_ad_token_provider: Optional[AzureADTokenProvider] = Field(
        default=None, description="Callback function to provide Azure Entra ID token."
    )

    _azure_ad_token: Any = PrivateAttr(default=None)
    _client: SyncAzureOpenAI = PrivateAttr()
    _aclient: AsyncAzureOpenAI = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-35-turbo",
        engine: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        reuse_client: bool = True,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
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
        # base class
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
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
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            timeout=timeout,
            reuse_client=reuse_client,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            azure_ad_token_provider=azure_ad_token_provider,
            use_azure_ad=use_azure_ad,
            api_version=api_version,
            callback_manager=callback_manager,
            http_client=http_client,
            async_http_client=async_http_client,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            **kwargs,
        )

    @model_validator(mode="before")
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

    def _get_client(self) -> SyncAzureOpenAI:
        if not self.reuse_client:
            return SyncAzureOpenAI(**self._get_credential_kwargs())

        if self._client is None:
            self._client = SyncAzureOpenAI(
                **self._get_credential_kwargs(),
            )
        return self._client

    def _get_aclient(self) -> AsyncAzureOpenAI:
        if not self.reuse_client:
            return AsyncAzureOpenAI(**self._get_credential_kwargs(is_async=True))

        if self._aclient is None:
            self._aclient = AsyncAzureOpenAI(
                **self._get_credential_kwargs(is_async=True),
            )
        return self._aclient

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
            import os

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

    @classmethod
    def class_name(cls) -> str:
        return "azure_openai_llm"
