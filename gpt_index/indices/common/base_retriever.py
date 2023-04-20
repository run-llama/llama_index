from abc import ABC, abstractmethod
from typing import List
from gpt_index.data_structs.node_v2 import NodeWithScore

from gpt_index.indices.query.schema import QueryBundle


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
        pass
