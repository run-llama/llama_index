from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.stepfun.com/v1"
DEFAULT_MODEL = "step-1v-8k"


class StepFun(OpenAILike):
    """
    The StepFun class is a subclass of OpenAILike and is used to interact with the StepFun model.

    Parameters
    ----------
        model (str): The name of the Stepfun model to use. See https://platform.stepfun.com/docs/llm/modeloverview for options.
        context_window (int): The maximum size of the context window for the model. See https://platform.stepfun.com/docs/llm/modeloverview for options.
        is_chat_model (bool): Indicates whether the model is a chat model.

    Attributes
    ----------
        model (str): The name of the Stepfun model to use.
        context_window (int): The maximum size of the context window for the model.
        is_chat_model (bool): Indicates whether the model is a chat model.

    """

    model: str = Field(
        description="The Stepfun model to use. See https://platform.stepfun.com/docs/llm/modeloverview for options."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model. See https://platform.stepfun.com/docs/llm/modeloverview for options.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI API client.

        Args:
            model (str): The name of the model to use. Defaults to DEFAULT_MODEL.
            temperature (float): The temperature to use for the model. Defaults to DEFAULT_TEMPERATURE.
            max_tokens (int): The maximum number of tokens to generate. Defaults to DEFAULT_NUM_OUTPUTS.
            additional_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to the model. Defaults to None.
            max_retries (int): The maximum number of retries to make when calling the API. Defaults to 5.
            api_base (Optional[str]): The base URL for the API. Defaults to DEFAULT_API_BASE.
            api_key (Optional[str]): The API key to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the model.

        Returns:
            None

        """
        additional_kwargs = additional_kwargs or {}

        api_base = get_from_param_or_env("api_base", api_base, "STEPFUN_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "STEPFUN_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Stpefun_LLM"
