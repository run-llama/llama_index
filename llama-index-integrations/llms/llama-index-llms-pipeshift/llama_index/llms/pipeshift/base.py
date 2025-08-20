import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.pipeshift.com/api/v0"


def validate_api_key_and_model(api_key: str, model: str) -> None:
    """
    Validate the API key and model name.

    Args:
        api_key (str): The API key to validate.
        model (str): The model name to validate.

    Raises:
        ValueError: If the API key or model name is invalid.

    """
    if not api_key:
        raise ValueError("Pipeshift API Key not found!")
    elif not isinstance(api_key, str) or len(api_key.strip()) == 0:
        raise ValueError("Invalid API key: API key must be a non-empty string.")

    if not model:
        raise ValueError("Model not specified. PLease enter model name")
    if not isinstance(model, str) or len(model.strip()) == 0:
        raise ValueError("Invalid model name: Model name must be a non-empty string.")


class Pipeshift(OpenAILike):
    """
    Pipeshift LLM.

    Examples:
        `pip install llama-index-llms-pipeshift`

        ```python
        from llama_index.llms.pipeshift import Pipeshift

        # set api key in env or in llm
        # import os
        # os.environ["PIPESHIFT_API_KEY"] = "your api key"

        llm = Pipeshift(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key="your_api_key"
        )

        resp = llm.complete("How fast is porsche gt3 rs?")
        print(resp)
        ```

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("PIPESHIFT_API_KEY", None)
        try:
            validate_api_key_and_model(api_key, model)
            super().__init__(
                model=model,
                api_key=api_key,
                api_base=api_base,
                is_chat_model=is_chat_model,
                **kwargs,
            )
        except ValueError as e:
            raise ValueError(e)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Pipeshift"
