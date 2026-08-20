import os
from typing import Any, Literal, Optional

from llama_index.llms.openai_like import OpenAILike
from pydantic import Field


class Cerebras(OpenAILike):
    """
    Cerebras LLM.

    Examples:
        `pip install llama-index-llms-cerebras`

        ```python
        from llama_index.llms.cerebras import Cerebras

        # Set up the Cerebras class with the required model and API key
        llm = Cerebras(model="llama-3.3-70b", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Why is fast inference important?")

        print(response)
        ```

    """

    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="Exact completion-token limit sent to the Cerebras API.",
    )

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = os.environ.get("CEREBRAS_BASE_URL", None)
        or "https://api.cerebras.ai/v1/",
        is_chat_model: bool = True,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high", "xhigh"]
        ] = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("CEREBRAS_API_KEY", None)

        assert api_key is not None, (
            "API Key not specified! Please set `CEREBRAS_API_KEY`!"
        )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        if self.max_completion_tokens is not None:
            model_kwargs.pop("max_tokens", None)
            model_kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = self.reasoning_effort
        return model_kwargs

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Cerebras"
