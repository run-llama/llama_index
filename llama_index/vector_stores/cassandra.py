"""Cassandra / Astra DB Vector store index.

An index based on a DB table with vector search capabilities,
powered by the cassIO library

"""

import logging
from typing import Any, List, Optional, cast


from llama_index.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.indices.query.embedding_utils import (
    get_top_k_mmr_embeddings,
)

from llama_index.vector_stores.types import (
    # MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

_logger = logging.getLogger(__name__)

DEFAULT_MMR_PREFETCH_FACTOR = 4.0


class CassandraVectorStore(VectorStore):
    """Cassandra Vector Store.

    An abstraction of a Cassandra table with
    vector-similarity-search. Documents, and their embeddings, are stored
    in a Cassandra table and a vector-capable index is used for searches.
    The table does not need to exist beforehand: if necessary it will
    be created behind the scenes.

    All Cassandra operations are done through the cassIO library.

    Args:
        session (cassandra.cluster.Session): the Cassandra session to use
        keyspace (str): name of the Cassandra keyspace to work in
        table (str): table name to use. If not existing, it will be created.
        embedding_dimension (int): length of the embedding vectors in use.
        ttl_seconds (Optional[int]): expiration time for inserted entries.
            Default is no expiration.

    """

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        session: Any,
        keyspace: str,
        table: str,
        embedding_dimension: int,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        import_err_msg = "`cassio` package not found, please run `pip install cassio`"
        try:
            import cassio  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self._session = session
        self._keyspace = keyspace
        self._table = table
        self._embedding_dimension = embedding_dimension
        self._ttl_seconds = ttl_seconds

        _logger.debug("Creating VectorTable")
        self.vector_table = cassio.vector.VectorTable(
            session=self._session,
            keyspace=self._keyspace,
            table=self._table,
            embedding_dimension=self._embedding_dimension,
            primary_key_type="TEXT",
        )

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeWithEmbedding]: list of embedding results

        """
        (document_ids0, documents, document_metadatas, embeddings,) = zip(
            *(
                (
                    result.id,
                    # The name of the key containing the text is a constant, so:
                    getattr(result.node, DEFAULT_TEXT_KEY, ""),
                    node_to_metadata_dict(
                        result.node,
                        remove_text=False,  # must keep the text here for retrieval
                        flat_metadata=self.flat_metadata,
                    ),
                    result.embedding,
                )
                for result in embedding_results
            )
        )
        document_ids = list(document_ids0)

        # TODO: batching or concurrent inserts
        _logger.debug(f"Adding {len(document_ids)} rows to table")
        for (doc_id, doc, md, emb) in zip(
            document_ids, documents, document_metadatas, embeddings
        ):
            self.vector_table.put(
                document=doc,
                embedding_vector=emb,
                document_id=doc_id,
                metadata=md,
                ttl_seconds=self._ttl_seconds,
            )

        return document_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        _logger.debug("Deleting a row from VectorTable")
        self.vector_table.delete(
            document_id=ref_doc_id,
        )

    @property
    def client(self) -> Any:
        """Return the underlying cassIO vector table object"""
        return self.vector_table

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Supported query modes: 'default' (most similar vectors) and 'mmr'.

        Args:
            query (VectorStoreQuery): the basic query definition. Defines:
                mode (VectorStoreQueryMode): one of the supported modes
                query_embedding (List[float]): query embedding to search against
                similarity_top_k (int): top k most similar nodes
                mmr_threshold (Optional[float]): this is the 0-to-1 MMR lambda.
                    If present, takes precedence over the kwargs parameter.
                    Ignored unless for MMR queries.

        Args for query.mode == 'mmr' (ignored otherwise):
            mmr_threshold (Optional[float]): this is the 0-to-1 lambda for MMR.
                Note that in principle mmr_threshold could come in the query
            mmr_prefetch_factor (Optional[float]): factor applied to top_k
                for prefetch pool size. Defaults to 4.0
            mmr_prefetch_k (Optional[int]): prefetch pool size. This cannot be
                passed together with mmr_prefetch_factor

        """
        _available_query_modes = [
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.MMR,
        ]
        if query.mode not in _available_query_modes:
            raise NotImplementedError(f"Query mode {query.mode} not available.")
        #
        query_embedding = cast(List[float], query.query_embedding)

        _logger.debug(f"Running ANN search on VectorTable (query mode: {query.mode})")
        if query.mode == VectorStoreQueryMode.DEFAULT:
            matches = self.vector_table.search(
                embedding_vector=query_embedding,
                top_k=query.similarity_top_k,
                metric="cos",
                metric_threshold=None,
            )
            top_k_scores = [match["distance"] for match in matches]
        elif query.mode == VectorStoreQueryMode.MMR:
            # Querying a larger number of vectors and then doing MMR on them.
            if (
                kwargs.get("mmr_prefetch_factor") is not None
                and kwargs.get("mmr_prefetch_k") is not None
            ):
                raise ValueError(
                    "'mmr_prefetch_factor' and 'mmr_prefetch_k' "
                    "cannot coexist in a call to query()"
                )
            else:
                if kwargs.get("mmr_prefetch_k") is not None:
                    prefetch_k0 = int(kwargs["mmr_prefetch_k"])
                else:
                    prefetch_k0 = int(
                        query.similarity_top_k
                        * kwargs.get("mmr_prefetch_factor", DEFAULT_MMR_PREFETCH_FACTOR)
                    )
            prefetch_k = max(prefetch_k0, query.similarity_top_k)
            #
            prefetch_matches = self.vector_table.search(
                embedding_vector=query_embedding,
                top_k=prefetch_k,
                metric="cos",
                metric_threshold=None,  # this is not `mmr_threshold`
            )
            #
            mmr_threshold = query.mmr_threshold or kwargs.get("mmr_threshold")
            if prefetch_matches:
                pf_match_indices, pf_match_embeddings = zip(
                    *enumerate(match["embedding_vector"] for match in prefetch_matches)
                )
            else:
                pf_match_indices, pf_match_embeddings = [], []
            pf_match_indices = list(pf_match_indices)
            pf_match_embeddings = list(pf_match_embeddings)
            mmr_similarities, mmr_indices = get_top_k_mmr_embeddings(
                query_embedding,
                pf_match_embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=pf_match_indices,
                mmr_threshold=mmr_threshold,
            )
            #
            matches = [prefetch_matches[mmr_index] for mmr_index in mmr_indices]
            top_k_scores = mmr_similarities

        top_k_nodes = []
        top_k_ids = []
        for match in matches:
            node = metadata_dict_to_node(match["metadata"])
            top_k_nodes.append(node)
            top_k_ids.append(match["document_id"])

        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            similarities=top_k_scores,
            ids=top_k_ids,
        )
