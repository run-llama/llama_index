from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph.sub_retrievers.base import (
    BaseLPGRetriever,
)
from llama_index.core.graph_stores.types import LabelledPropertyGraphStore
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.settings import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle


class LPGVectorRetriever(BaseLPGRetriever):
    def __init__(
        self,
        graph_store: LabelledPropertyGraphStore,
        include_text: bool = True,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: int = 10,
        **kwargs: Any
    ) -> None:
        self._retriever_kwargs = kwargs or {}
        self._embed_model = embed_model or Settings.embed_model
        self._similarity_top_k = similarity_top_k

        super().__init__(graph_store=graph_store, include_text=include_text, **kwargs)

    def _get_vector_store_query(self, query_bundle: QueryBundle) -> VectorStoreQuery:
        if query_bundle.embedding is None:
            query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(
                query_bundle.embedding_strs
            )

        return VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            top_k=self._similarity_top_k,
            **self._retriever_kwargs,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_store_query = self._get_vector_store_query(query_bundle)

        # TODO: this is not the proper return type yet
        nodes = self._graph_store.vector_query(vector_store_query)

        return self._parse_results(nodes)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_store_query = self._get_vector_store_query(query_bundle)

        # TODO: this is not the proper return type yet
        nodes = await self._graph_store.avector_query(vector_store_query)

        return await self._aparse_results(nodes)
