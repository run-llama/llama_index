import os
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
import requests
from openai import OpenAI as SyncOpenAI
from openai import AsyncOpenAI



DEFAULT_PLAYGROUND_MODEL = "nv-rerank-qa-mistral-4b:1"
BASE_PLAYGROUND_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
DEFAULT_TOP_N = 2


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
    """NVIDIA's API Catalog Reranker Connector"""
    model: str = Field(
                    default=DEFAULT_PLAYGROUND_MODEL,
                    description="The NVIDIA API Catalog reranker to use.",
                )
    top_n: int = Field(
                    default=2,
                    description="The default value for top_n is 2",
                )
    url: Optional[str] =Field(
                    default=BASE_PLAYGROUND_URL,
                    description="The default API Catalog reranker's url",
                ),
    _api_key: Any = PrivateAttr()
    def __init__(
        self,
        model: str = DEFAULT_PLAYGROUND_MODEL,
        top_n: int = DEFAULT_TOP_N,
        timeout: float = 120,
        max_retries: int = 5,
        api_key: Optional[str] = None,
        url : str = BASE_PLAYGROUND_URL,
    ) -> None:      

        self._api_key = get_from_param_or_env(
            "api_key", api_key, "NVIDIA_API_KEY", ""
        )

        if not self._api_key:
            raise ValueError(
                "The NVIDIA API key must be provided as an environment variable or as a parameter."
            )
               
        super().__init__(top_n=top_n, model=model, url=url)
        


    @classmethod
    def class_name(cls) -> str:
        return "NVIDIAReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        try:
            assert query_bundle is not None
        except :            
            raise ValueError("Missing query bundle in extra info. Please do not give empty query!")
        if len(nodes) == 0:
            return []
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

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
            texts = [{"text": node.node.get_content()} for node in nodes]
            
            payloads = {
                "model": self.model,
                "query": {"text": query_bundle.query_str},
                "passages":texts,
            }
            
            response = session.post(self.url, headers=headers, json=payloads)

            try :
                assert response.status_code == 200                
                
                results = response.json()["rankings"]
                new_nodes = []
            
                for result in results:
                    new_node_with_score = NodeWithScore(
                        node=nodes[result['index']].node, score=result['logit']
                    )
                    new_nodes.append(new_node_with_score)
                new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                    : self.top_n
                ]
                event.on_end(payload={EventPayload.NODES: new_nodes})
            except :
                print(f"Query unsuccessful {response.status_code}, please visit this page for more info : https://developer.nvidia.com/docs/nemo-microservices/inference/tools.html")

        return new_nodes