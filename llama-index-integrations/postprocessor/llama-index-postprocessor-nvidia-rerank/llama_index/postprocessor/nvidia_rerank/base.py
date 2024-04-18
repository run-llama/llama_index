import os
from typing import Any, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
import requests
from openai import OpenAI as SyncOpenAI
from openai import AsyncOpenAI
from enum import Enum


class ModelType(Enum):
    NVIDIAAPICatalog = "catalog"
    NIM = "nim"



DEFAULT_PLAYGROUND_MODEL = "nv-rerank-qa-mistral-4b:1" 
BASE_PLAYGROUND_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

DEFAULT_TOP_N = 2


class NVIDIARerank(BaseNodePostprocessor):
    """NVIDIA's API Catalog Reranker Connector"""
    model: Optional[str] = Field(
                    default=DEFAULT_PLAYGROUND_MODEL,
                    description="The NVIDIA API Catalog reranker to use.",
                )
    top_n: Optional[int] = Field(
                    default=2,
                    description="The default value for top_n is 2",
                )
    url: Optional[str] =Field(
                    default=BASE_PLAYGROUND_URL,
                    description="The default API Catalog reranker's url",
                ),
    type_mode : Optional[str] = Field(
                    default="catalog" ,
                    description="Default to NVIDIA API Catalog reranker to use, support mode switching",
                )
    _api_key: Any = PrivateAttr()

    def __init__(
        self,
        type_mode : str = "catalog" ,
        model: str = DEFAULT_PLAYGROUND_MODEL,
        top_n: int = DEFAULT_TOP_N,
        timeout: float = 120,
        max_retries: int = 5,
        api_key: Optional[str] = None,
        url : str = BASE_PLAYGROUND_URL,
    ) :
        
        super().__init__(top_n=top_n, model=model, url=url)
        self.model = model 
        self.top_n = top_n
        self.type_mode = type_mode
        self.url = url
        if type_mode=='nim':
            self.mode(type_mode=type_mode)
        

    def mode(self, type_mode=type_mode, base_url = url , model=model , top_n = top_n):
        if isinstance(self, str):
            raise ValueError("Please construct the model before calling mode()")
        out = self
        if type_mode in ["nvidia", "catalog"]:
            key_var = "NVIDIA_API_KEY"
            api_key = os.getenv(key_var)
            
            if not api_key.startswith("nvapi-"):
                raise ValueError(f"No {key_var} in env/fed as api_key. (nvapi-...)")
        if type_mode in ["openai"]:
            key_var = "OPENAI_API_KEY"
            if not api_key or not api_key.startswith("sk-"):
                api_key = os.getenv(key_var) 
            if not api_key.startswith("sk-"):
                raise ValueError(f"No {key_var} in env/fed as api_key. (sk-...)")

        out.type_mode = type_mode
        

        if type_mode == "catalog":
            ## NVIDIA API Catalog Integration: OpenAPI-spec gateway over NVCF endpoints
            out.top_n = top_n
            out.url = base_url 
            ## API Catalog is early, so no models list yet. Undercut to nvcf for now.
            out.model = model
            out._api_key = api_key
            

        elif type_mode == "nim":
            ## OpenAPI-style specs to connect to NeMo Inference Microservices etc.
            ## Most generic option, requires specifying base_url            
            out.top_n = top_n
            out.url = base_url +'/ranking'
            ## API Catalog is early, so no models list yet. Undercut to nvcf for now.
            out.model = model

        else:
            options = ["catalog", "nim"]
            raise ValueError(f"Unknown mode: `{type_mode}`. Expected one of {options}.")       

        return out
    
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
        model =self.model.default
        
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
            texts = [{"text": node.node.get_content()} for node in nodes]
            
            payloads = {
                "model": model,
                "query": {"text": query_bundle.query_str},
                "passages":texts,
            }
            if self.type_mode =='nim':
                current_url=self.url
                print(f"model {model} , url :{current_url}")
                response=requests.post(current_url,json=payloads)
            elif self.type_mode =='catalog':
                current_url = self.url[0].default
                print(f"model {model} , url :{current_url}")
                headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            }
                response = session.post(current_url, headers=headers, json=payloads)
            else:
                raise ValueError(
                f"Currently there are only 2 backends supported, default backend is catalog , your current backend is {self.type_mode}"                )    

            
            try :
                assert response.status_code == 200                
                
                results = response.json()["rankings"]
                print("----------> \n", results)
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