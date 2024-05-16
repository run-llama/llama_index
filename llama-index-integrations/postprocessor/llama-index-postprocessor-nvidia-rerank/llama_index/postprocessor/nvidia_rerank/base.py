from typing import Any, List, Optional, Literal, Generator

from urllib.parse import urlparse
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

from llama_index.core.base.llms.generic_utils import get_from_param_or_env


DEFAULT_MODEL = "nv-rerank-qa-mistral-4b:1"
DEFAULT_BASE_URL = "https://ai.api.nvidia.com/v1"

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
    _api_key: str = PrivateAttr("API_KEY_NOT_PROVIDED")  # TODO: should be SecretStr
    _mode: str = PrivateAttr("nvidia")
    _base_url: str = PrivateAttr(DEFAULT_BASE_URL)

    def _set_api_key(self, nvidia_api_key: str = None, api_key: str = None) -> None:
        self._api_key = get_from_param_or_env(
            "api_key",
            nvidia_api_key or api_key,
            "NVIDIA_API_KEY",
            "API_KEY_NOT_PROVIDED",
        )

    def __init__(
        self,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._set_api_key(nvidia_api_key, api_key)

    @property
    def available_models(self) -> List[Model]:
        """Get available models."""
        # there is one model on ai.nvidia.com and available as a local NIM
        ids = [DEFAULT_MODEL]
        return [Model(id=id) for id in ids]

    def mode(
        self,
        mode: Literal["nvidia", "nim"] = "nvidia",
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "NVIDIARerank":
        """
        Change the mode.

        There are two modes, "nvidia" and "nim". The "nvidia" mode is the default mode
        and is used to interact with hosted NVIDIA NIMs. The "nim" mode is
        used to interact with local NVIDIA NIM endpoints, which are typically hosted
        on-premises.

        For the "nvidia" mode, the "api_key" parameter is available to specify your
        API key. If not specified, the NVIDIA_API_KEY environment variable will be used.

        For the "nim" mode, the "base_url" is required and "model" is recommended. Set
        base_url to the url of your NVIDIA NIM endpoint. For instance,
        "https://localhost:1976/v1", it should end in "/v1". Additionally, the "model"
        parameter must be set to the name of the model inside the NIM.
        """
        if isinstance(self, str):
            raise ValueError("Please construct the model before calling mode()")

        if mode == "nim":
            if not base_url:
                raise ValueError("base_url is required for nim mode")
        if not base_url:
            base_url = DEFAULT_BASE_URL

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
            self._set_api_key(api_key)

        return self

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIARerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatch_event = dispatcher.get_dispatch_event()
        dispatch_event(
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
                if self._mode == "nvidia":
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

        dispatch_event(ReRankEndEvent(nodes=results))
        return results
