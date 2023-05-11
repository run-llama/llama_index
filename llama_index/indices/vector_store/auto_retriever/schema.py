

from dataclasses import dataclass
from typing import List

from dataclasses_json import DataClassJsonMixin


@dataclass
class MetadataInfo(DataClassJsonMixin):
    name: str
    type: str
    description: str

@dataclass
class VectorStoreInfo(DataClassJsonMixin):
    metadata_info: List[MetadataInfo]
    content_info: str

