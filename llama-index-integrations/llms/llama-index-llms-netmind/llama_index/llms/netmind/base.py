import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


class NetmindLLM(OpenAILike):
    """
    Netmind LLM.

    Examples:
        `pip install llama-index-llms-netmind`

        ```python
        from llama_index.llms.netmind import NetmindLLM

        # set api key in env or in llm
        # import os
        # os.environ["NETMIND_API_KEY"] = "your api key"

        llm = NetmindLLM(
            model="meta-llama/Llama-3.3-70B-Instruct", api_key="your_api_key"
        )

        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.netmind.ai/inference-api/openai/v1",
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("NETMIND_API_KEY", None)
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
        return "NetmindLLM"
