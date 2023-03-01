"""Elasticsearch/Opensearch vector store."""
import json
from typing import Any, Dict, List, Optional

from gpt_index.data_structs import Node
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
)


class OpensearchVectorClient:
    """
    Object encapsulating an Opensearch index that has vector search enabled.

    If the index does not yet exist, it is created during init.
    Therefore, The underlying index is assumed to either:
    1) not exist yet or 2) be created due to previous usage of this class.

    Args:
        endpoint (str): URL (http/https) of elasticsearch endpoint
        index (str): Name of the elasticsearch index
        dim (int): Dimension of the vector
        embedding_field (str): Name of the field in the index to store
            embedding array in.
        text_field (str): Name of the field to grab text from
        method (Optional[dict]): Opensearch "method" JSON obj for configuring
            the KNN index.
            This includes engine, metric, and other config params. Defaults to:
                {
                    "name": "hnsw",
                    "space_type": "l2",
                    "engine": "faiss",
                    "parameters": {"ef_construction": 256, "m": 48}
                }
    """

    def __init__(
        self,
        endpoint: str,
        index: str,
        dim: int,
        embedding_field: str = "embedding",
        text_field: str = "content",
        method: Optional[dict] = None,
    ):
        """Init params."""
        if method is None:
            method = {
                "name": "hnsw",
                "space_type": "l2",
                "engine": "nmslib",
                "parameters": {"ef_construction": 256, "m": 48},
            }
        import_err_msg = "`httpx` package not found, please run `pip install httpx`"
        if embedding_field is None:
            embedding_field = "embedding"
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        self._embedding_field = embedding_field
        self._client = httpx.Client(base_url=endpoint)
        self._endpoint = endpoint
        self._dim = dim
        self._index = index
        self._text_field = text_field
        # initialize mapping
        idx_conf = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
            "mappings": {
                "properties": {
                    embedding_field: {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": method,
                    },
                }
            },
        }
        res = self._client.put(f"/{self._index}", json=idx_conf)
        # will 400 if the index already existed, so allow 400 errors right here
        assert res.status_code == 200 or res.status_code == 400

    def index_results(self, results: List[NodeEmbeddingResult]) -> List[str]:
        """Store results in the index."""
        bulk_req: List[Dict[Any, Any]] = []
        for result in results:
            bulk_req.append({"index": {"_index": self._index, "_id": result.id}})
            bulk_req.append(
                {
                    self._text_field: result.node.text,
                    self._embedding_field: result.embedding,
                }
            )
        bulk = "\n".join([json.dumps(v) for v in bulk_req]) + "\n"
        res = self._client.post(
            "/_bulk", headers={"Content-Type": "application/x-ndjson"}, content=bulk
        )
        assert res.status_code == 200
        assert not res.json()["errors"], "expected no errors while indexing docs"
        return [r.id for r in results]

    def delete_doc_id(self, doc_id: str) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id
        """
        self._client.delete(f"{self._index}/_doc/{doc_id}")

    def do_approx_knn(
        self, query_embedding: List[float], k: int
    ) -> VectorStoreQueryResult:
        """Do approximate knn."""
        res = self._client.post(
            f"{self._index}/_search",
            json={
                "size": k,
                "query": {
                    "knn": {self._embedding_field: {"vector": query_embedding, "k": k}}
                },
            },
        )
        nodes = []
        ids = []
        scores = []
        for hit in res.json()["hits"]["hits"]:
            source = hit["_source"]
            text = source[self._text_field]
            doc_id = hit["_id"]
            node = Node(text=text, extra_info=source, doc_id=doc_id)
            ids.append(doc_id)
            nodes.append(node)
            scores.append(hit["_score"])
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=scores)


class OpensearchVectorStore(VectorStore):
    """Elasticsearch/Opensearch vector store.

    Args:
        client (OpensearchVectorClient): Vector index client to use
            for data insertion/querying.

    """

    stores_text: bool = True

    def __init__(
        self,
        client: OpensearchVectorClient,
    ) -> None:
        """Initialize params."""
        import_err_msg = "`httpx` package not found, please run `pip install httpx`"
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ValueError(import_err_msg)
        self._client = client

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    @property
    def config_dict(self) -> dict:
        """Get config dict."""
        return {}

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to index.

        Args
            embedding_results: List[NodeEmbeddingResult]: list of embedding results

        """
        self._client.index_results(embedding_results)
        return [result.id for result in embedding_results]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        self._client.delete_doc_id(doc_id)

    def query(
        self, query_embedding: List[float], similarity_top_k: int
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes

        """
        return self._client.do_approx_knn(query_embedding, similarity_top_k)
