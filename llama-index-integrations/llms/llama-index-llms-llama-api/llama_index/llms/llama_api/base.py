from llama_index.core.base.llms.types import (
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.llms.openai_like import OpenAILike


class LlamaAPI(OpenAILike):
    """
    LlamaAPI LLM.

    Examples:
        `pip install llama-index-llms-llama-api`

        ```python
        from llama_index.llms.llama_api import LlamaAPI

        # Obtain an API key from https://www.llama-api.com/
        api_key = "your-api-key"

        llm = LlamaAPI(model="llama3.1-8b", context_window=128000, is_function_calling_model=True, api_key=api_key)

        # Call the complete method with a prompt
        resp = llm.complete("Paul Graham is ")

        print(resp)
        ```

    """

    model: str = Field(
        default="llama3.1-8b",
        description=LLMMetadata.model_fields["model_name"].description,
    )

    api_base: str = Field(
        default="https://api.llmapi.com/",
        description="The base URL for OpenAI API.",
    )

    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.model_fields["is_function_calling_model"].description,
    )

    @classmethod
    def class_name(cls) -> str:
        return "llama_api_llm"
