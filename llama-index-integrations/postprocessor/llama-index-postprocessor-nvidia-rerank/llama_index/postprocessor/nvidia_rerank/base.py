import os
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
import requests


DEFAULT_MODEL = "nv-rerank-qa-mistral-4b:1" 
BASE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

DEFAULT_TOP_N = 2
 
model_lookup = {"nvidia":[DEFAULT_MODEL], "nim":[DEFAULT_MODEL]}


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
    model: Optional[str] = Field(
                    default=DEFAULT_MODEL,
                    description="The NVIDIA API Catalog reranker to use.",
                )
    top_n: Optional[int] = Field(
                    default=2,
                    description="The default value for top_n is 2",
                )
    url: Optional[str] =Field(
                    default=BASE_URL,
                    description="The default API Catalog reranker's url",
                ),
    _mode : Optional[str] = Field(
                    default="nvidia" ,
                    description="Default to NVIDIA API Catalog reranker to use, support mode switching",
                )
    _api_key: Any = PrivateAttr()
    _mode : Any = PrivateAttr()
    _headers : Any = PrivateAttr()

    def __init__(
        self,
        _mode : Optional[str]=None ,
        model: str = DEFAULT_MODEL,
        top_n: int = DEFAULT_TOP_N,      
        url : str = BASE_URL,
        _api_key: Optional[str] = None,
    ) : 
        
        super().__init__(top_n=top_n, model=model, url=url)
        self.model = model 
        self.top_n = top_n
        self._api_key = None
        self.url = url
        self._mode = None
        self._headers = None
        
        
        
    def get_available_models():
        return model_lookup.items()

    def mode(self, mode :str = None, base_url :str = url , model :str =model , top_n : int = top_n , api_key :str = None):
        if isinstance(self, str):
            raise ValueError("Please construct the model before calling mode()")
        out = self
        if mode in ["nvidia"]:
            key_var = "NVIDIA_API_KEY"
            my_key = get_from_param_or_env("api_key", api_key, "NVIDIA_API_KEY", "")

            #api_key = os.getenv(key_var)
            
            if not api_key.startswith("nvapi-"):
                raise ValueError(f"No {key_var} in env/fed as api_key. (nvapi-...)")
        
        out._mode = mode
        

        if mode == "nvidia":
            ## NVIDIA API Catalog Integration: OpenAPI-spec gateway over NVCF endpoints
            out.top_n = top_n
            out.url = base_url[0].default 
            ## API Catalog is early, so no models list yet. Undercut to nvcf for now.
            out.model = model_lookup[mode][0]
            out._api_key = my_key
            out._headers = {
            "Authorization": f"Bearer {my_key}",
            "Accept": "application/json",
            }
            

        elif mode == "nim":
            ## OpenAPI-style specs to connect to NeMo Inference Microservices etc.
            ## Most generic option, requires specifying base_url            
            out.top_n = top_n
            if base_url.endswith('/ranking'):
                raise ValueError(f"Incorrect url format {base_url}, you do not need to extend '/ranking' at the end, as an example, here is a valid url format :http://.../v1/")
            out.url = base_url +'/ranking'
            ## API Catalog is early, so no models list yet. Undercut to nvcf for now.
            out.model = model_lookup[mode][0]
            out._headers = None

        else:
            options = ["nvidia", "nim"]
            raise ValueError(f"Unknown mode: `{mode}`. Expected one of {options}.")       

        return out
    
    @classmethod
    def class_name(cls) -> str:
        return "NVIDIAReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info. Please do not give empty query!")
        if len(nodes) == 0:
            return []
        model =self.model
        
        top_n=self.top_n
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
            
            payloads = {
                "model": model,
                "query": {"text": query_bundle.query_str},
                "passages":[{"text": node.node.get_content()} for node in nodes],
            }
            
            current_url = self.url            
            response = session.post(current_url, headers=self._headers, json=payloads)
            response.raise_for_status()        

            
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