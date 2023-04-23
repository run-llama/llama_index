from typing import List
from gpt_index.data_structs.node_v2 import NodeWithScore
from gpt_index.indices.base_retriever import BaseRetriever
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle


class TransformRetriever(BaseRetriever):
    def __init__(
        self,
        retriever: BaseRetriever,
        query_transform: BaseQueryTransform,
    ) -> None:
        self._retriever = retriever
        self._query_transform = query_transform

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_bundle = self._query_transform.run(query_bundle)
        return self._retriever.retrieve(query_bundle)
