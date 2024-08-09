from typing import Any, List, Optional, Literal, Generator

from urllib.parse import urlparse, urlunparse
from llama_index.core.bridge.pydantic import Field, PrivateAttr, BaseModel
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
from deprecated import deprecated
from llama_index.core.base.llms.generic_utils import get_from_param_or_env


DEFAULT_MODEL = "nv-rerank-qa-mistral-4b:1"
BASE_URL = "https://ai.api.nvidia.com/v1"

MODEL_ENDPOINT_MAP = {
    DEFAULT_MODEL: BASE_URL,
    "nvidia/nv-rerankqa-mistral-4b-v3": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
}

KNOWN_URLS = list(MODEL_ENDPOINT_MAP.values())

dispatcher = get_dispatcher(__name__)


class Model(BaseModel):
    id: str


class NVIDIARerank(BaseNodePostprocessor):
    """NVIDIA's API Catalog Reranker Connector."""

    class Config:
        validate_assignment = True

    model: Optional[str] = Field(
        default=DEFAULT_MODEL,
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
    _api_key: str = PrivateAttr("NO_API_KEY_PROVIDED")  # TODO: should be SecretStr
    _mode: str = PrivateAttr("nvidia")
    _is_hosted: bool = PrivateAttr(True)
    _base_url: str = PrivateAttr(BASE_URL)

    def _set_api_key(self, nvidia_api_key: str = None, api_key: str = None) -> None:
        self._api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
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
            **kwargs: Additional keyword arguments.

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY` environment variable.
        """
        super().__init__(model=model, **kwargs)

        if base_url is None or base_url in MODEL_ENDPOINT_MAP.values():
            base_url = MODEL_ENDPOINT_MAP.get(model, BASE_URL)
        else:
            base_url = self._validate_url(base_url)

        self._api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        self._is_hosted = self._base_url in KNOWN_URLS

        if self._is_hosted and self._api_key == "NO_API_KEY_PROVIDED":
            warnings.warn(
                "An API key is required for hosted NIM. This will become an error in 0.2.0."
            )

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
            raise ValueError(
                f"Invalid base_url, Expected format is 'http://host:port': {base_url}"
            )
        if result.path:
            normalized_path = result.path.strip("/")
            if normalized_path == "v1":
                pass
            elif normalized_path == "v1/rankings":
                warnings.warn(f"{expected_format} Rest is Ignored.")
            else:
                raise ValueError(f"Base URL path is not recognized. {expected_format}")
        return urlunparse((result.scheme, result.netloc, "v1", "", "", ""))

    @property
    def available_models(self) -> List[Model]:
        """Get available models."""
        # all available models are in the map
        ids = MODEL_ENDPOINT_MAP.keys()
        return [Model(id=id) for id in ids]

    @deprecated(
        version="0.1.2",
        reason="Will be removed in 0.2. Construct with `base_url` instead.",
    )
    def mode(
        self,
        mode: Literal["nvidia", "nim"] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIARerank":
        """
        Deprecated: use NVIDIARerank(base_url=...) instead.
        """
        if isinstance(self, str):
            raise ValueError("Please construct the model before calling mode()")

        self._is_hosted = mode == "nvidia"

        if not self._is_hosted:
            if not base_url:
                raise ValueError("base_url is required for nim mode")
        else:
            api_key = get_from_param_or_env("api_key", api_key, "NVIDIA_API_KEY")
        if not base_url:
            base_url = BASE_URL

        self._mode = mode
        if base_url:
            # TODO: change this to not require /v1 at the end. the current
            #       implementation is for consistency, but really this code
            #       should dictate which version it works with
            components = urlparse(base_url)
            if not components.scheme or not components.netloc:
                raise ValueError(
                    f"Incorrect url format, use https://host:port/v1, given '{base_url}'"
                )
            last_nonempty_path_component = [x for x in components.path.split("/") if x][
                -1
            ]
            if last_nonempty_path_component != "v1":
                raise ValueError(
                    f"Incorrect url format, use https://host:post/v1 ending with /v1, given '{base_url}'"
                )
            self._base_url = base_url
        if model:
            self.model = model
        if api_key:
            self._api_key = api_key

        return self

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
                    "query": {"text": query_bundle.query_str},
                    "passages": [
                        {"text": n.get_content(metadata_mode=MetadataMode.EMBED)}
                        for n in batch
                    ],
                }
                # the hosted NIM path is different from the local NIM path
                url = self._base_url
                if self._is_hosted:
                    if url.endswith("/v1"):
                        url += "/retrieval/nvidia/reranking"
                else:
                    url += "/ranking"
                response = session.post(url, headers=_headers, json=payloads)
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
