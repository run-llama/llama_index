from llama_index.core.base.llms.types import (
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.llms.openai_like.base import OpenAILike


class OPEA(OpenAILike):
    """
    Adapter for a OPEA LLM.

    Examples:
        `pip install llama-index-llms-opea`

        ```python
        from llama_index.llms.opea import OPEA

        llm = OPEA(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_base="http://localhost:8080/v1",
        )
        ```

    """

    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )

    @classmethod
    def class_name(cls) -> str:
        return "OPEA"
