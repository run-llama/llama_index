from typing import Any, Dict

from pydantic import root_validator

from llama_index.llms.openai import OpenAI


class AzureOpenAI(OpenAI):
    """
    Azure OpenAI

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

    model: str  # override default specified in OpenAI
    engine: str  # model deployment name

    @root_validator()
    def validate_env(cls, values: Dict) -> Dict:
        """Validate necessary environment variables are set."""
        try:
            import openai

            if openai.api_base == "https://api.openai.com/v1":
                raise ValueError(
                    "You must set OPENAI_API_BASE to your Azure endpoint. "
                    "It should look like https://YOUR_RESOURCE_NAME.openai.azure.com/"
                )
            if openai.api_type not in ("azure", "azure_ad", "azuread"):
                raise ValueError(
                    "You must set OPENAI_API_TYPE to one of "
                    "(`azure`, `azuread`, `azure_ad`) for Azure OpenAI."
                )
            if openai.api_version is None:
                raise ValueError("You must set OPENAI_API_VERSION for Azure OpenAI.")
        except ImportError:
            raise ImportError(
                "You must install the `openai` package to use Azure OpenAI."
            )
        return values

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        model_kwargs = super()._model_kwargs
        model_kwargs.pop("model")
        model_kwargs["engine"] = self.engine
        return model_kwargs
