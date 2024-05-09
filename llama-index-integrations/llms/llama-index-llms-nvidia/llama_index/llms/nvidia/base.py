from typing import (
    Any,
    Optional,
    List,
    Literal,
)

from llama_index.core.bridge.pydantic import PrivateAttr, BaseModel
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)

from llama_index.llms.nvidia.utils import API_CATALOG_MODELS

from llama_index.llms.openai_like import OpenAILike

DEFAULT_MODEL = "meta/llama3-8b-instruct"
BASE_URL = "https://integrate.api.nvidia.com/v1/"


class Model(BaseModel):
    id: str


class NVIDIA(OpenAILike):
    """NVIDIA's API Catalog Connector."""

    _mode: str = PrivateAttr("nvidia")

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = 1024,
        **kwargs: Any,
    ) -> None:
        api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=BASE_URL,
            max_tokens=max_tokens,
            is_chat_model=True,
            default_headers={"User-Agent": "llama-index-llms-nvidia"},
            **kwargs,
        )

    @property
    def available_models(self) -> List[Model]:
        ids = API_CATALOG_MODELS.keys()
        if self._mode == "nim":
            ids = [model.id for model in self._get_client().models.list()]
        return [Model(id=name) for name in ids]

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIA"

    def mode(
        self,
        mode: Optional[Literal["nvidia", "nim"]] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIA":
        """
        Change the mode.

        There are two modes, "nvidia" and "nim". The "nvidia" mode is the default
        mode and is used to interact with hosted NIMs. The "nim" mode is used to
        interact with NVIDIA NIM endpoints, which are typically hosted on-premises.

        For the "nvidia" mode, the "api_key" parameter is available to specify
        your API key. If not specified, the NVIDIA_API_KEY environment variable
        will be used.

        For the "nim" mode, the "base_url" parameter is required and the "model"
        parameter may be necessary. Set base_url to the url of your local NIM
        endpoint. For instance, "https://localhost:9999/v1". Additionally, the
        "model" parameter must be set to the name of the model inside the NIM.
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
