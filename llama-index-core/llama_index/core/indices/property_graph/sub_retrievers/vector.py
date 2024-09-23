from typing import Any, List, Sequence, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph.sub_retrievers.base import (
    BasePGRetriever,
)
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    KG_SOURCE_REL,
    VECTOR_SOURCE_KEY,
)
from llama_index.core.settings import Settings
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    MetadataFilters,
)


class VectorContextRetriever(BasePGRetriever):
    """A retriever that uses a vector store to retrieve nodes based on a query.

    Args:
        graph_store (PropertyGraphStore):
            The graph store to retrieve data from.
        include_text (bool, optional):
            Whether to include source text in the retrieved nodes. Defaults to True.
        embed_model (Optional[BaseEmbedding], optional):
            The embedding model to use. Defaults to Settings.embed_model.
        vector_store (Optional[BasePydanticVectorStore], optional):
            The vector store to use. Defaults to None.
            Should be supplied if the graph store does not support vector queries.
        similarity_top_k (int, optional):
            The number of top similar kg nodes to retrieve. Defaults to 4.
        path_depth (int, optional):
            The depth of the path to retrieve for each node. Defaults to 1 (i.e. a triple).
        similarity_score (float, optional):
            The minimum similarity score to retrieve the nodes. Defaults to None.
    """

    def __init__(
        self,
        graph_store: PropertyGraphStore,
        include_text: bool = True,
        include_properties: bool = False,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        similarity_score: Optional[float] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> None:
        self._retriever_kwargs = kwargs or {}
        self._embed_model = embed_model or Settings.embed_model
        self._similarity_top_k = similarity_top_k
        self._vector_store = vector_store
        self._path_depth = path_depth
        self._similarity_score = similarity_score
        self._filters = filters

        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    def _get_vector_store_query(self, query_bundle: QueryBundle) -> VectorStoreQuery:
        if query_bundle.embedding is None:
            query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(
                query_bundle.embedding_strs
            )

        return VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            filters=self._filters,
            **self._retriever_kwargs,
        )

    def _get_kg_ids(self, kg_nodes: Sequence[BaseNode]) -> List[str]:
        """Backward compatibility method to get kg_ids from kg_nodes."""
        return [node.metadata.get(VECTOR_SOURCE_KEY, node.id_) for node in kg_nodes]

    async def _aget_vector_store_query(
        self, query_bundle: QueryBundle
    ) -> VectorStoreQuery:
        if query_bundle.embedding is None:
            query_bundle.embedding = (
                await self._embed_model.aget_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )

        return VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            filters=self._filters,
            **self._retriever_kwargs,
        )

    def retrieve_from_graph(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_store_query = self._get_vector_store_query(query_bundle)

        triplets = []
        kg_ids = []
        new_scores = []
        if self._graph_store.supports_vector_queries:
            result = self._graph_store.vector_query(vector_store_query)
            if len(result) != 2:
                raise ValueError("No nodes returned by vector_query")
            kg_nodes, scores = result

            kg_ids = [node.id for node in kg_nodes]
            triplets = self._graph_store.get_rel_map(
                kg_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
            )

        elif self._vector_store is not None:
            query_result = self._vector_store.query(vector_store_query)
            if query_result.nodes is not None and query_result.similarities is not None:
                kg_ids = self._get_kg_ids(query_result.nodes)
                scores = query_result.similarities
                kg_nodes = self._graph_store.get(ids=kg_ids)
                triplets = self._graph_store.get_rel_map(
                    kg_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
                )

            elif query_result.ids is not None and query_result.similarities is not None:
                kg_ids = query_result.ids
                scores = query_result.similarities
                kg_nodes = self._graph_store.get(ids=kg_ids)
                triplets = self._graph_store.get_rel_map(
                    kg_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
                )

        for triplet in triplets:
            score1 = (
                scores[kg_ids.index(triplet[0].id)] if triplet[0].id in kg_ids else 0.0
            )
            score2 = (
                scores[kg_ids.index(triplet[2].id)] if triplet[2].id in kg_ids else 0.0
            )
            new_scores.append(max(score1, score2))

        assert len(triplets) == len(new_scores)

        # filter by similarity score
        if self._similarity_score:
            filtered_data = [
                (triplet, score)
                for triplet, score in zip(triplets, new_scores)
                if score >= self._similarity_score
            ]
            # sort by score
            top_k = sorted(filtered_data, key=lambda x: x[1], reverse=True)
        else:
            # sort by score
            top_k = sorted(zip(triplets, new_scores), key=lambda x: x[1], reverse=True)

        return self._get_nodes_with_score([x[0] for x in top_k], [x[1] for x in top_k])

    async def aretrieve_from_graph(
        self, query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        vector_store_query = await self._aget_vector_store_query(query_bundle)

        triplets = []
        kg_ids = []
        new_scores = []
        if self._graph_store.supports_vector_queries:
            result = await self._graph_store.avector_query(vector_store_query)
            if len(result) != 2:
                raise ValueError("No nodes returned by vector_query")

            kg_nodes, scores = result
            kg_ids = [node.id for node in kg_nodes]
            triplets = await self._graph_store.aget_rel_map(
                kg_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
            )

        elif self._vector_store is not None:
            query_result = await self._vector_store.aquery(vector_store_query)
            if query_result.nodes is not None and query_result.similarities is not None:
                kg_ids = self._get_kg_ids(query_result.nodes)
                scores = query_result.similarities
                kg_nodes = await self._graph_store.aget(ids=kg_ids)
                triplets = await self._graph_store.aget_rel_map(
                    kg_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
                )

            elif query_result.ids is not None and query_result.similarities is not None:
                kg_ids = query_result.ids
                scores = query_result.similarities
                kg_nodes = await self._graph_store.aget(ids=kg_ids)
                triplets = await self._graph_store.aget_rel_map(
                    kg_nodes, depth=self._path_depth, ignore_rels=[KG_SOURCE_REL]
                )

        for triplet in triplets:
            score1 = (
                scores[kg_ids.index(triplet[0].id)] if triplet[0].id in kg_ids else 0.0
            )
            score2 = (
                scores[kg_ids.index(triplet[2].id)] if triplet[2].id in kg_ids else 0.0
            )
            new_scores.append(max(score1, score2))

        assert len(triplets) == len(new_scores)

        # filter by similarity score
        if self._similarity_score:
            filtered_data = [
                (triplet, score)
                for triplet, score in zip(triplets, new_scores)
                if score >= self._similarity_score
            ]
            # sort by score
            top_k = sorted(filtered_data, key=lambda x: x[1], reverse=True)
        else:
            # sort by score
            top_k = sorted(zip(triplets, new_scores), key=lambda x: x[1], reverse=True)

        return self._get_nodes_with_score([x[0] for x in top_k], [x[1] for x in top_k])
