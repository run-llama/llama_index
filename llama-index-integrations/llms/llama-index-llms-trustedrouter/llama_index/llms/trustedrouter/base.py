import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


DEFAULT_API_BASE = "https://api.trustedrouter.com/v1"
DEFAULT_MODEL = "trustedrouter/zdr"


class TrustedRouter(OpenAILike):
    """
    TrustedRouter LLM class.

    TrustedRouter (https://trustedrouter.com) is an OpenAI-compatible LLM
    router that serves many models behind one endpoint.

    Examples:
        `pip install llama-index-llms-trustedrouter`

        ```python
        from llama_index.llms.trustedrouter import TrustedRouter

        # set api key in env or in llm
        # import os
        # os.environ["TRUSTEDROUTER_API_KEY"] = "your api key"

        llm = TrustedRouter(
            model="trustedrouter/zdr", api_key="your_api_key"
        )

        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("TRUSTEDROUTER_API_KEY", None)
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
        return "TrustedRouter"
