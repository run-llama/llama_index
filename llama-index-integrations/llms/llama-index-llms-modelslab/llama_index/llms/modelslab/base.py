import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

MODELSLAB_API_BASE = "https://modelslab.com/uncensored-chat/v1"


class ModelsLabLLM(OpenAILike):
    """
    ModelsLab LLM integration for LlamaIndex.

    Provides uncensored Llama 3.1 language models via ModelsLab's
    OpenAI-compatible API. Suitable for RAG pipelines, agents, and
    workflows requiring unrestricted language generation with a
    128K token context window.

    Models:
        - ``llama-3.1-8b-uncensored`` — fast, efficient (default)
        - ``llama-3.1-70b-uncensored`` — higher quality, deeper reasoning

    Examples:
        ``pip install llama-index-llms-modelslab``

        ```python
        from llama_index.llms.modelslab import ModelsLabLLM

        # Set MODELSLAB_API_KEY env var or pass api_key directly
        llm = ModelsLabLLM(
            model="llama-3.1-8b-uncensored",
            api_key="your-modelslab-api-key",
        )

        resp = llm.complete("Explain transformers in simple terms.")
        print(resp)
        ```

        Use in a RAG pipeline::

        ```python
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        from llama_index.llms.modelslab import ModelsLabLLM
        from llama_index.core import Settings

        Settings.llm = ModelsLabLLM(model="llama-3.1-70b-uncensored")

        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query("What is the main topic?")
        ```

    Get your API key at: https://modelslab.com
    API docs: https://docs.modelslab.com/uncensored-chat

    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-uncensored",
        api_key: Optional[str] = None,
        api_base: str = MODELSLAB_API_BASE,
        is_chat_model: bool = True,
        is_function_calling_model: bool = False,
        context_window: int = 131072,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("MODELSLAB_API_KEY")
        if not api_key:
            raise ValueError(
                "ModelsLab API key not found. "
                "Set the MODELSLAB_API_KEY environment variable or pass api_key directly. "
                "Get your key at https://modelslab.com"
            )
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            context_window=context_window,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ModelsLabLLM"
