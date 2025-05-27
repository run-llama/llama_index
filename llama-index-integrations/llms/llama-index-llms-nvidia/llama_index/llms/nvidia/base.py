from typing import Any, Optional, List, Literal, Union, Dict, TYPE_CHECKING
from deprecated import deprecated
import warnings
import json
import os

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)

from llama_index.core.llms.llm import ToolSelection
from .utils import (
    BASE_URL,
    DEFAULT_MODEL,
    MODEL_TABLE,
    CHAT_MODEL_TABLE,
    Model,
    determine_model,
)

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


KNOWN_URLS = {
    model.endpoint if model.endpoint else None for model in MODEL_TABLE.values()
}


class NVIDIA(OpenAILike, FunctionCallingLLM):
    """NVIDIA's API Catalog Connector."""

    _mode: str = PrivateAttr(default="nvidia")
    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _is_hosted: bool = PrivateAttr(True)

    def __init__(
        self,
        model: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = os.getenv("NVIDIA_BASE_URL", BASE_URL),
        max_tokens: Optional[int] = 1024,
        **kwargs: Any,
    ) -> None:
        """
        Initialize an instance of the NVIDIA class.

        This class provides an interface to the NVIDIA NIM. By default, it connects to a hosted NIM,
        but you can switch to an on-premises NIM by providing a `base_url`.

        Args:
            model (str, optional): The model to use for the NIM.
            nvidia_api_key (str, optional): The API key for the NVIDIA NIM. Defaults to None.
            api_key (str, optional): An alternative parameter for providing the API key. Defaults to None.
            base_url (str, optional): The base URL for the NIM. Use this to switch to an on-premises NIM.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            **kwargs: Additional keyword arguments.

        API Keys:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY` environment variable.

        Raises:
            DeprecationWarning: If an API key is not provided for a hosted NIM, a warning is issued. This will become an error in version 0.2.0.

        """
        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        base_url = base_url or BASE_URL

        super().__init__(
            api_key=api_key,
            api_base=base_url,
            max_tokens=max_tokens,
            default_headers={"User-Agent": "llama-index-llms-nvidia"},
            **kwargs,
        )
        self.model = model
        self._is_hosted = base_url in KNOWN_URLS or base_url == BASE_URL
        if self._is_hosted and api_key == "NO_API_KEY_PROVIDED":
            warnings.warn(
                "An API key is required for the hosted NIM. This will become an error in 0.2.0.",
            )
        if not self.model:
            if self._is_hosted:
                self.model = DEFAULT_MODEL
            else:
                self.__get_default_model()

        if not self.model.startswith("nvdev/"):
            self._validate_model(self.model)  ## validate model
        self.is_chat_model = self._is_chat_model()
        self.is_function_calling_model = self._is_function_calling_model()

    def __get_default_model(self):
        """Set default model."""
        if not self._is_hosted:
            valid_models = [
                model.id
                for model in self.available_models
                if not model.base_model or model.base_model == model.id
            ]
            self.model = next(iter(valid_models), None)
            if self.model:
                warnings.warn(
                    f"Default model is set as: {self.model}. \n"
                    "Set model using model parameter. \n"
                    "To get available models use available_models property.",
                    UserWarning,
                )
            else:
                raise ValueError("No locally hosted model was found.")
        else:
            self.model = DEFAULT_MODEL

    def _validate_model(self, model_name: str) -> None:
        """
        Validates compatibility of the hosted model with the client.

        Args:
            model_name (str): The name of the model.

        Raises:
            ValueError: If the model is incompatible with the client.

        """
        if self._is_hosted:
            if model := determine_model(model_name):
                if not model.client:
                    warnings.warn(f"Unable to determine validity of {model.id}")
                elif model.client != self.class_name():
                    raise ValueError(
                        f"Model {model.id} is incompatible with client {self.class_name()}. "
                        f"Please check `{self.class_name()}.get_available_models`"
                    )
                if model.endpoint:
                    self.api_base = model.endpoint
            else:
                candidates = [
                    model for model in self.available_models if model.id == model_name
                ]
                assert len(candidates) <= 1, (
                    f"Multiple candidates for {model_name} "
                    f"in `available_models`: {candidates}"
                )
                if candidates:
                    model = candidates[0]
                    warnings.warn(
                        f"Found {model_name} in available_models, but type is "
                        "unknown and inference may fail."
                    )
                else:
                    if model_name.startswith("nvdev/"):  # assume valid
                        model = Model(id=model_name)
                    else:
                        raise ValueError(
                            f"Model {model_name} is unknown, check `available_models`"
                        )
        else:
            if model_name not in [model.id for model in self.available_models]:
                raise ValueError(f"No locally hosted {model_name} was found.")

    @property
    def available_models(self) -> List[Model]:
        models = []
        for element in self._get_client().models.list().data:
            if not (model := determine_model(element.id)):
                model = Model(id=element.id)
            models.append(model)
        # only exclude models in hosted mode. in non-hosted mode, the administrator has control
        # over the model name and may deploy an excluded name that will work.
        if self._is_hosted:
            exclude = {
                "mistralai/mixtral-8x22b-v0.1",  # not a /chat/completion endpoint
            }
            models = [model for model in models if model.id not in exclude]
        return models

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIA"

    @deprecated(
        version="0.1.3",
        reason="Will be removed in 0.2. Construct with `base_url` instead.",
    )
    def mode(
        self,
        mode: Optional[Literal["nvidia", "nim"]] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIA":
        """
        Deprecated: use NVIDIA(base_url="...") instead.
        """
        if mode == "nim":
            if not base_url:
                raise ValueError("base_url is required for nim mode")
        if mode == "nvidia":
            api_key = get_from_param_or_env(
                "api_key",
                api_key,
                "NVIDIA_API_KEY",
            )
            base_url = base_url or BASE_URL

        self._mode = mode
        if base_url:
            self.api_base = base_url
        if model:
            self.model = model
        if api_key:
            self.api_key = api_key

        return self

    def _is_chat_model(self):
        model = determine_model(self.model)
        return model and model.id in CHAT_MODEL_TABLE

    def _is_function_calling_model(self):
        model = determine_model(self.model)
        return model and model.supports_tools

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare the chat with tools."""
        # misralai uses the same openai tool format
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            # if not isinstance(tool_call, ToolCall):
            #     raise ValueError("Invalid tool_call object")

            argument_dict = json.loads(tool_call.function.arguments)

            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections
