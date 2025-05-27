import os
from typing import Any, Optional, Dict

from llama_index.llms.openai_like import OpenAILike


class Databricks(OpenAILike):
    """
    Databricks LLM.

    Examples:
        `pip install llama-index-llms-databricks`

        ```python
        from llama_index.llms.databricks import Databricks

        # Set up the Databricks class with the required model, API key and serving endpoint
        llm = Databricks(model="databricks-dbrx-instruct", api_key="your_api_key", api_base="https://[your-work-space].cloud.databricks.com/serving-endpoints")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of open source LLMs")

        print(response)
        ```

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("DATABRICKS_TOKEN", None)
        api_base = api_base or os.environ.get("DATABRICKS_SERVING_ENDPOINT", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Databricks"

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Get the kwargs that need to be provided to the model invocation."""
        # Fix the input to work with the Databricks API
        if "tool_choice" in kwargs and "tools" not in kwargs:
            del kwargs["tool_choice"]

        return super()._get_model_kwargs(**kwargs)
