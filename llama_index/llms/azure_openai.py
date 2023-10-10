from typing import Any, Dict, Optional

from llama_index.bridge.pydantic import Field, root_validator
from llama_index.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_utils import resolve_from_aliases

AZURE_OPENAI_API_TYPE = "azure"


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
    - `OPENAI_API_TYPE`: set this to `azure`, `azure_ad`, or `azuread`
    - `OPENAI_API_VERSION`: set this to `2023-05-15`
        This may change in the future.
    - `OPENAI_API_BASE`: your endpoint should look like the following
        https://YOUR_RESOURCE_NAME.openai.azure.com/
    - `OPENAI_API_KEY`: your API key

    More information can be found here:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python
    """

    engine: str = Field(description="The name of the deployed azure engine.")

    def __init__(
        self,
        model: str = "gpt-35-turbo",
        engine: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        api_key: Optional[str] = None,
        api_type: Optional[str] = AZURE_OPENAI_API_TYPE,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        # aliases for engine
        deployment_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        deployment: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        engine = resolve_from_aliases(
            engine,
            deployment_name,
            deployment_id,
            deployment,
        )

        if engine is None:
            raise ValueError("You must specify an `engine` parameter.")

        super().__init__(
            engine=engine,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            api_key=api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            callback_manager=callback_manager,
            **kwargs,
        )

    @root_validator
    def validate_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate necessary credentials are set."""
        if values["api_base"] == "https://api.openai.com/v1":
            raise ValueError(
                "You must set OPENAI_API_BASE to your Azure endpoint. "
                "It should look like https://YOUR_RESOURCE_NAME.openai.azure.com/"
            )
        if values["api_type"] not in ("azure", "azure_ad", "azuread"):
            raise ValueError(
                "You must set OPENAI_API_TYPE to one of "
                "(`azure`, `azuread`, `azure_ad`) for Azure OpenAI."
            )
        if values["api_version"] is None:
            raise ValueError("You must set OPENAI_API_VERSION for Azure OpenAI.")

        return values

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        model_kwargs = super()._model_kwargs
        model_kwargs.pop("model")
        model_kwargs["engine"] = self.engine
        return model_kwargs

    @classmethod
    def class_name(cls) -> str:
        return "azure_openai_llm"
