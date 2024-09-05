from typing import Any, List, Optional, Generator, Literal


from urllib.parse import urlparse, urljoin
from llama_index.core.bridge.pydantic import Field, PrivateAttr, BaseModel, ConfigDict
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
import requests
import warnings
from llama_index.core.base.llms.generic_utils import get_from_param_or_env


DEFAULT_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"

MODEL_ENDPOINT_MAP = {
    "nvidia/nv-rerankqa-mistral-4b-v3": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
    "nv-rerank-qa-mistral-4b:1": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
}

dispatcher = get_dispatcher(__name__)


class Model(BaseModel):
    id: str
    base_model: Optional[str] = None


class NVIDIARerank(BaseNodePostprocessor):
    """NVIDIA's API Catalog Reranker Connector."""

    model_config = ConfigDict(validate_assignment=True)
    model: Optional[str] = Field(
        description="The NVIDIA API Catalog reranker to use.",
    )
    top_n: Optional[int] = Field(
        default=5,
        ge=0,
        description="The number of nodes to return.",
    )
    max_batch_size: Optional[int] = Field(
        default=64,
        ge=1,
        description="The maximum batch size supported by the inference server.",
    )
    truncate: Optional[Literal["NONE", "END"]] = Field(
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is model dependent and is likely to raise error if an "
            "input is too long."
        ),
        default=None,
    )
    _api_key: str = PrivateAttr("NO_API_KEY_PROVIDED")  # TODO: should be SecretStr
    _mode: str = PrivateAttr("nvidia")
    _is_hosted: bool = PrivateAttr(True)
    _base_url: str = PrivateAttr(MODEL_ENDPOINT_MAP.get(DEFAULT_MODEL))
    _inference_url: Optional[str] = PrivateAttr(None)

    def _set_api_key(self, nvidia_api_key: str = None, api_key: str = None) -> None:
        self._api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

    def __init__(
        self,
        model: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize a NVIDIARerank instance.

        This class provides access to a NVIDIA NIM for reranking. By default, it connects to a hosted NIM, but can be configured to connect to an on-premises NIM using the `base_url` parameter. An API key is required for hosted NIM.

        Args:
            model (str): The model to use for reranking.
            nvidia_api_key (str, optional): The NVIDIA API key. Defaults to None.
            api_key (str, optional): The API key. Defaults to None.
            base_url (str, optional): The base URL of the on-premises NIM. Defaults to None.
            truncate (str): "NONE", "END", truncate input text if it exceeds
                            the model's context length. Default is model dependent and
                            is likely to raise an error if an input is too long.
            **kwargs: Additional keyword arguments.

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY` environment variable.
        """
        if not base_url or (base_url in MODEL_ENDPOINT_MAP.values() and not model):
            model = model or DEFAULT_MODEL
        super().__init__(model=model, **kwargs)

        base_url = base_url or MODEL_ENDPOINT_MAP.get(DEFAULT_MODEL)
        self._is_hosted = base_url in MODEL_ENDPOINT_MAP.values()

        if not self._is_hosted and base_url:
            self._base_url = base_url.rstrip("/") + "/"

        self._api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        if not self._is_hosted:  # on-premises mode
            # in this case we trust the model name and base_url
            self._inference_url = self._validate_url(base_url) + "/ranking"
        else:  # hosted mode
            if self._api_key == "NO_API_KEY_PROVIDED":
                raise ValueError("An API key is required for hosted NIM.")
            if not model:
                model = MODEL_ENDPOINT_MAP.get(base_url)
            if model in MODEL_ENDPOINT_MAP:
                self._inference_url = MODEL_ENDPOINT_MAP[model]

        self.model = model
        if self._is_hosted and not self.model:
            self.model = DEFAULT_MODEL
        elif not self._is_hosted and not self.model:
            self.__get_default_model()

        self._validate_model(self.model)  ## validate model

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

    def _get_models(self) -> List[Model]:
        session = requests.Session()

        if self._is_hosted:
            _headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            }
        else:
            _headers = {
                "Accept": "application/json",
            }
        url = (
            "https://integrate.api.nvidia.com/v1/models"
            if self._is_hosted
            else urljoin(self._base_url, "models")
        )
        response = session.get(url, headers=_headers)
        response.raise_for_status()

        assert (
            "data" in response.json()
        ), "Response does not contain expected 'data' key"
        assert isinstance(
            response.json()["data"], list
        ), "Response 'data' is not a list"
        assert all(
            isinstance(result, dict) for result in response.json()["data"]
        ), "Response 'data' is not a list of dictionaries"
        assert all(
            "id" in result for result in response.json()["data"]
        ), "Response 'rankings' is not a list of dictionaries with 'id'"

        return [
            Model(
                id=model["id"],
                base_model=getattr(model, "params", {}).get("root", None),
            )
            for model in response.json()["data"]
        ]

    def _validate_url(self, base_url):
        """
        Base URL Validation.
        ValueError : url which do not have valid scheme and netloc.
        Warning : v1/rankings routes.
        ValueError : Any other routes other than above.
        """
        expected_format = "Expected format is 'http://host:port'."
        result = urlparse(base_url)
        if not (result.scheme and result.netloc):
            raise ValueError(f"Invalid base_url, {expected_format}")
        if result.path:
            normalized_path = result.path.strip("/")
            if normalized_path == "v1":
                pass
            elif normalized_path == "v1/rankings":
                warnings.warn(f"{expected_format} Rest is Ignored.")
            else:
                raise ValueError(f"Invalid base_url, {expected_format}")
        return base_url

    def _validate_model(self, model_name: str) -> None:
        """
        Validates compatibility of the hosted model with the client.

        Args:
            model_name (str): The name of the model.

        Raises:
            ValueError: If the model is incompatible with the client.
        """
        if self._is_hosted:
            if model_name not in MODEL_ENDPOINT_MAP:
                raise ValueError(
                    f"Model {model_name} is incompatible with client {self.class_name()}. "
                    f"Please check `{self.class_name()}.available_models()`."
                )
        else:
            if model_name not in [model.id for model in self.available_models]:
                raise ValueError(f"No locally hosted {model_name} was found.")

    @property
    def available_models(self) -> List[Model]:
        """Get available models."""
        # all available models are in the map
        ids = MODEL_ENDPOINT_MAP.keys()
        if not self._is_hosted:
            return self._get_models()
        else:
            return [Model(id=id) for id in ids]

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIARerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model,
            )
        )

        if query_bundle is None:
            raise ValueError(
                "Missing query bundle in extra info. Please do not give empty query!"
            )
        if len(nodes) == 0:
            return []

        session = requests.Session()

        _headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        # TODO: replace with itertools.batched in python 3.12
        def batched(ls: list, size: int) -> Generator[List[NodeWithScore], None, None]:
            for i in range(0, len(ls), size):
                yield ls[i : i + size]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            results = []
            for batch in batched(nodes, self.max_batch_size):
                payloads = {
                    "model": self.model,
                    **({"truncate": self.truncate} if self.truncate else {}),
                    "query": {"text": query_bundle.query_str},
                    "passages": [
                        {"text": n.get_content(metadata_mode=MetadataMode.EMBED)}
                        for n in batch
                    ],
                }
                response = session.post(
                    self._inference_url, headers=_headers, json=payloads
                )
                response.raise_for_status()
                # expected response format:
                # {
                #     "rankings": [
                #         {
                #             "index": 0,
                #             "logit": 0.0
                #         },
                #         ...
                #     ]
                # }
                assert (
                    "rankings" in response.json()
                ), "Response does not contain expected 'rankings' key"
                assert isinstance(
                    response.json()["rankings"], list
                ), "Response 'rankings' is not a list"
                assert all(
                    isinstance(result, dict) for result in response.json()["rankings"]
                ), "Response 'rankings' is not a list of dictionaries"
                assert all(
                    "index" in result and "logit" in result
                    for result in response.json()["rankings"]
                ), "Response 'rankings' is not a list of dictionaries with 'index' and 'logit' keys"
                for result in response.json()["rankings"][: self.top_n]:
                    results.append(
                        NodeWithScore(
                            node=batch[result["index"]].node, score=result["logit"]
                        )
                    )
            if len(nodes) > self.max_batch_size:
                results.sort(key=lambda x: x.score, reverse=True)
            results = results[: self.top_n]
            event.on_end(payload={EventPayload.NODES: results})

        dispatcher.event(ReRankEndEvent(nodes=results))
        return results
