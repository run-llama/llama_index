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
from llama_index.core.base.llms.generic_utils import get_from_param_or_env


DEFAULT_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"

MODEL_ENDPOINT_MAP = {
    "nvidia/nv-rerankqa-mistral-4b-v3": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
    "nv-rerank-qa-mistral-4b:1": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
}

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
    truncate: Optional[Literal["NONE", "END"]] = Field(
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is model dependent and is likely to raise error if an "
            "input is too long."
        ),
    )
    _api_key: str = PrivateAttr("NO_API_KEY_PROVIDED")  # TODO: should be SecretStr
    _mode: str = PrivateAttr("nvidia")
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
            truncate (str): "NONE", "END", truncate input text if it exceeds
                            the model's context length. Default is model dependent and
                            is likely to raise an error if an input is too long.
            **kwargs: Additional keyword arguments.

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY` environment variable.
        """
        super().__init__(model=model, **kwargs)

        self._api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "NO_API_KEY_PROVIDED",
        )

        if base_url:  # on-premises mode
            # in this case we trust the model name and base_url
            self._inference_url = self._validate_url(base_url) + "/rankings"
        else:  # hosted mode
            if model not in MODEL_ENDPOINT_MAP:
                raise ValueError(
                    f"Model '{model}' not found. "
                    f"Available models are: {', '.join(MODEL_ENDPOINT_MAP.keys())}"
                )
            if self._api_key == "NO_API_KEY_PROVIDED":
                raise ValueError("An API key is required for hosted NIM.")
            self._inference_url = MODEL_ENDPOINT_MAP[model]

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
