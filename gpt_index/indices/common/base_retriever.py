from abc import ABC, abstractmethod
from typing import List, Union
from gpt_index.data_structs.node_v2 import NodeWithScore

from gpt_index.indices.query.schema import QueryBundle


class BaseRetriever(ABC):
    def retrieve(
        self, str_or_query_bundle: Union[str, QueryBundle]
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._retrieve(str_or_query_bundle)

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        pass
