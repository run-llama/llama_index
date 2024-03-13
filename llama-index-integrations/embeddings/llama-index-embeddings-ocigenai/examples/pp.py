from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from pydantic import BaseModel
from llama_index.core.bridge.pydantic import Field, PrivateAttr



class AA(BaseModel):
    a: str = Field(description="Id of the OCI Generative AI embedding model to use.", default="cohere.embed-english-v3.0")
    b: Optional[str] = Field(
        description="Model Input type. If not provided, search_document and search_query are used when needed."
    )
    c: str = Field(description="Id of the OCI Generative AI embedding model to use.")
       
    def __init__(
        self,
        a: str ="jfj",
        b: Optional[str] = 'hhh',
        c: str = None,
              
    ):
        
        super().__init__(
            a=a,
            b=b,
            c=c
             
        )

aa=AA(a="cohere.embed-english-v3.0", b="START")
print(aa)