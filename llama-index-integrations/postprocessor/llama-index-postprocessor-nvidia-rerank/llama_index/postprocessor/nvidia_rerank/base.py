import os
from typing import Any, List, Optional, Iterable
from itertools import islice

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
import requests

# from _statics import MODEL_SPECS, Model


DEFAULT_MODEL = "nv-rerank-qa-mistral-4b:1"
BASE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

DEFAULT_TOP_N = 2
DEFAULT_BATCH_SIZE = 32

model_lookup = {"nvidia": [DEFAULT_MODEL], "nim": [DEFAULT_MODEL]}


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class NVIDIARerank(BaseNodePostprocessor):
    """NVIDIA's API Catalog Reranker Connector."""

    model: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description="The NVIDIA API Catalog reranker to use.",
    )
    top_n: Optional[int] = Field(
        default=2,
        description="The default value for top_n is 2",
    )
    max_batch_size: Optional[int] = Field(
        default=32,
        description="The default value for batch_size is 2",
    )
    _api_key: Any = PrivateAttr()
    _mode: str = PrivateAttr("nvidia")
    _headers: Any = PrivateAttr()
    _score: Any = PrivateAttr()
    _url: Any = PrivateAttr()

    def __init__(
        self,
        _mode: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        top_n: int = DEFAULT_TOP_N,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
        _api_key: Optional[str] = None,
    ):
        super().__init__(top_n=top_n, model=model)
        self.model = model
        self.top_n = top_n
        self._api_key = None
        self._url = None
        self._mode = None
        self._headers = None
        self._score = None

    def get_available_models():
        return model_lookup.items()

    def mode(
        self,
        mode: str = None,
        base_url: str = BASE_URL,
        model: str = model,
        api_key: str = None,
    ):
        if isinstance(self, str):
            raise ValueError("Please construct the model before calling mode()")
        out = self
        if mode in ["nvidia"]:
            key_var = "NVIDIA_API_KEY"
            my_key = get_from_param_or_env("api_key", api_key, "NVIDIA_API_KEY", "")

            # api_key = os.getenv(key_var)

            if not api_key.startswith("nvapi-"):
                raise ValueError(f"No {key_var} in env/fed as api_key. (nvapi-...)")

        out._mode = mode

        if mode == "nvidia":
            ## NVIDIA API Catalog Integration: OpenAPI-spec gateway over NVCF endpoints

            out._url = BASE_URL
            ## API Catalog is early, so no models list yet. Undercut to nvcf for now.
            out.model = model_lookup[mode][0]
            out._api_key = my_key
            out._headers = {
                "Authorization": f"Bearer {my_key}",
                "Accept": "application/json",
            }
            out._score = "logit"

        elif mode == "nim":
            ## OpenAPI-style specs to connect to NeMo Inference Microservices etc.
            ## Most generic option, requires specifying base_url

            if base_url.endswith("/ranking"):
                raise ValueError(
                    f"Incorrect url format {base_url}, you do not need to extend '/ranking' at the end, as an example, here is a valid url format :http://.../v1/"
                )
            out._url = base_url + "/ranking"
            out._headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
            }
            out._score = "score"

            ## API Catalog is early, so no models list yet. Undercut to nvcf for now.
            out.model = model_lookup[mode][0]

        else:
            options = ["nvidia", "nim"]
            raise ValueError(f"Unknown mode: `{mode}`. Expected one of {options}.")

        return out

    @classmethod
    def class_name(cls) -> str:
        return "NVIDIAReranker"

    def _batch(self, iterable: Iterable, size: int):
        """Batch an iterable into chunks of a given size."""
        iterator = iter(iterable)
        for first in iterator:
            yield list(islice(iterator, size))

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError(
                "Missing query bundle in extra info. Please do not give empty query!"
            )
        if len(nodes) == 0:
            return []
        model = self.model

        top_n = self.top_n
        session = requests.Session()

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            current_url = self._url

            new_nodes = []
            results = []

            for batch_nodes in self._batch(nodes, self.max_batch_size):
                payloads = {
                    "model": model,
                    "query": {"text": query_bundle.query_str},
                    "passages": [{"text": n.get_content()} for n in batch_nodes],
                }
                response = session.post(
                    current_url, headers=self._headers, json=payloads
                )

                response.raise_for_status()
                results = response.json()["rankings"]
                for result in results:
                    new_node_with_score = NodeWithScore(
                        node=nodes[result["index"]], score=result[self._score]
                    )
                    new_nodes.append(new_node_with_score)
            if len(nodes) > self.max_batch_size:
                new_nodes = sorted(new_nodes, key=lambda x: -x.score if x.score else 0)
            new_nodes = new_nodes[: self.top_n]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
