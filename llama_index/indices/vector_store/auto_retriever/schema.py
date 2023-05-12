

from typing import List

from pydantic import BaseModel


class MetadataInfo(BaseModel):
    name: str
    type: str
    description: str

class VectorStoreInfo(BaseModel):
    metadata_info: List[MetadataInfo]
    content_info: str

