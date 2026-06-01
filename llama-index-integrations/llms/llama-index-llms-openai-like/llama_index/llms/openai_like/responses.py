from typing import Any, Optional, Union

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai.responses import OpenAIResponses, Tokenizer


class OpenAILikeResponses(OpenAIResponses):
    """
    OpenAILikeResponses LLM.

    A thin wrapper around the OpenAI Responses API that makes it compatible with
    3rd party tools that provide an openai-compatible Responses API endpoint.

    Args:
        model (str):
            The model to use for the api.
        api_base (str):
            The base url to use for the api.
            Defaults to "https://api.openai.com/v1".
        api_key (str):
            The api key to use for the api.
            Set this to some random string if your API does not require an api key.
        context_window (int):
            The context window to use for the api. Set this to your model's context window for the best experience.
            Defaults to 3900.
        is_function_calling_model (bool):
            Whether the model supports OpenAI function calling/tools over the API.
            Defaults to False.
        tokenizer (Tokenizer or str or None):
            An instance of a tokenizer object that has an encode method, or the name
            of a tokenizer model from Hugging Face. If left as None, then this
            disables inference of max_tokens.
        max_output_tokens (int):
            The max number of tokens to generate.
            Defaults to None.
        temperature (float):
            The temperature to use for the api.
            Default is 0.1.
        additional_kwargs (dict):
            Specify additional parameters to the request body.
        max_retries (int):
            How many times to retry the API call if it fails.
            Defaults to 3.
        timeout (float):
            How long to wait, in seconds, for an API call before failing.
            Defaults to 60.0.
        default_headers (dict):
            Override the default headers for API requests.
            Defaults to None.
        http_client (httpx.Client):
            Pass in your own httpx.Client instance.
            Defaults to None.
        async_http_client (httpx.AsyncClient):
            Pass in your own httpx.AsyncClient instance.
            Defaults to None.

    Examples:
        `pip install llama-index-llms-openai-like`

        ```python
        from llama_index.llms.openai_like import OpenAILikeResponses

        llm = OpenAILikeResponses(
            model="my model",
            api_base="https://hostname.com/v1",
            api_key="fake",
            context_window=128000,
            is_function_calling_model=True,
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

    context_window: Optional[int] = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.model_fields["context_window"].description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_function_calling_model"].description,
    )
    tokenizer: Union[Tokenizer, str, None] = Field(
        default=None,
        description=(
            "An instance of a tokenizer object that has an encode method, or the name"
            " of a tokenizer model from Hugging Face. If left as None, then this"
            " disables inference of max_tokens."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if isinstance(self.tokenizer, str):
            try:
                import transformers  # noqa: F401
            except ImportError:
                raise ImportError(
                    "The `transformers` package is required when passing a string "
                    "tokenizer name. Install it with: "
                    "`pip install llama-index-llms-openai-like[transformers]`"
                )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window or DEFAULT_CONTEXT_WINDOW,
            num_output=self.max_output_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        if isinstance(self.tokenizer, str):
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.tokenizer)
        return self.tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILikeResponses"
