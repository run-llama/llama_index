import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike
from pydantic import Field
from llama_index.core.llms.callbacks import (
    llm_completion_callback,
)
from llama_index.core.base.llms.types import (
    CompletionResponse,
    MessageRole,
    ChatMessage,
)


from contextual import ContextualAI


class Contextual(OpenAILike):
    """
    Generate a response using Contextual's Grounded Language Model (GLM), an LLM engineered specifically to prioritize faithfulness to in-context retrievals over parametric knowledge to reduce hallucinations in Retrieval-Augmented Generation.

    The total request cannot exceed 32,000 tokens. Email glm-feedback@contextual.ai with any feedback or questions.

    Examples:
        `pip install llama-index-llms-contextual`

        ```python
        from llama_index.llms.contextual import Contextual

        # Set up the Contextual class with the required model and API key
        llm = Contextual(model="contextual-clm", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```
    """

    model: str = Field(description="The model to use. Currently only supports `v1`.")
    api_key: str = Field(
        description="The API key to use.", default=os.environ.get("API_KEY", None)
    )
    base_url: str = Field(
        description="The base URL to use.", default="https://api.contextual.com"
    )
    system_prompt: str = Field(
        description="Instructions that the model follows when generating responses. Note that we do not guarantee that the model follows these instructions exactly."
    )
    avoid_commentary: bool = Field(
        description="Flag to indicate whether the model should avoid providing additional commentary in responses. Commentary is conversational in nature and does not contain verifiable claims; therefore, commentary is not strictly grounded in available context. However, commentary may provide useful context which improves the helpfulness of responses.",
        default=False,
    )
    client: Any = Field(default=None, exclude=True, description="Contextual AI Client")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.contextual.com",
        system_prompt: str = "",
        avoid_commentary: bool = False,
        **openai_llm_kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("API_KEY", None)

        try:
            self.client = ContextualAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise ValueError(f"Failed to initialize Contextual client: {e}")

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=base_url,
            is_chat_model=openai_llm_kwargs.pop("is_chat_model", True),
            **openai_llm_kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "contextual-clm"

    # Synchronous Methods
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): The input prompt to generate completion for.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            str: The generated text completion.
        """
        return self._generate(
            knowledge=None,
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            model=self.model,
            system_prompt=self.system_prompt,
            **kwargs,
        )

    def _generate(
        self, knowledge, messages, model, system_prompt, **kwargs
    ) -> CompletionResponse:
        """
        Generate completion for the given prompt.
        """
        raw_message = self.client.generate.create(
            messages=messages,
            knowledge=knowledge,
            model=self.model_name,
            system_prompt=system_prompt,
            extra_body={
                "avoid_commentary": self.avoid_commentary,
            },
        )
