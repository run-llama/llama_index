from typing import Any, Dict

from llama_index.llms.openai import OpenAI


class AzureOpenAI(OpenAI):
    """
    Azure OpenAI

    Args:
    - model: Name of the model (e.g. `text-davinci-003`)
        This in only used to decide completion vs. chat endpoint.
    - engine: This will correspond to the custom name you chose
        for your deployment when you deployed a model.

    You must have the following environment variables set:
    - `OPENAI_API_TYPE`: set this to `azure`
    - `OPENAI_API_VERSION`: set this to `2023-05-15`
        This may change in the future.
    - `OPENAI_API_BASE`: your endpoint should look like the following
        https://YOUR_RESOURCE_NAME.openai.azure.com/
    - `OPENAI_API_KEY`: your API key

    More information can be found here:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?tabs=command-line&pivots=programming-language-python
    """

    engine: str  # deployment name

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        openai_kwargs = super()._model_kwargs
        openai_kwargs.pop("model")
        openai_kwargs["engine"] = self.engine
