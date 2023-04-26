"""LanceDB vector store."""
from typing import Any, Dict, List, Optional, cast

# import numpy as np

from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
)


class LanceDBVectorStore(VectorStore):
    """The LanceDB Vector Store.

    Stores text and embeddings in LanceDB. The vector store will open an existing LanceDB dataset or create
    the dataset if it does not exist.

    Args:
        uri (str, required): Location where LanceDB will store its files.
        table_name (str, optional): The table name where the embeddings will be stored. Defaults to "vectors".
        nprobes (int, optional): The number of probes used. A higher number makes search more accurate but also slower.
            Defaults to 20.
        refine_factor: (int, optional): Refine the results by reading extra elements and re-ranking them in memory.
            Defaults to None

    Raises:
        ImportError: Unable to import `lancedb`.

    Returns:
        LanceDBVectorStore: VectorStore that supports creating LanceDB datasets and querying it.
    """

    stores_text = True

    def __init__(
        self,
        uri: str,
        table_name: str = "vectors",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import_err_msg = "`lancedb` package not found, please run `pip install lancedb`"
        try:
            import lancedb  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self.connection = lancedb.connect(uri)
        self.uri = uri
        self.table_name = table_name
        self.nprobes = nprobes
        self.refine_factor = refine_factor

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        return cls(**config_dict)

    @property
    def client(self) -> None:
        """Get client."""
        return None

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "uri": self.uri,
            "table_name": self.table_name,
            "nprobes": self.nprobes,
            "refine_factor": self.refine_factor,
        }

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        data = []
        ids = []
        for result in embedding_results:
            data.append(
                {
                    "id": result.id,
                    "doc_id": result.doc_id,
                    "vector": result.embedding,
                    "text": result.node.get_text(),
                }
            )
            ids.append(result.id)

        if self.table_name in self.connection.table_names():
            tbl = self.connection.open_table(self.table_name)
            tbl.add(data)
        else:
            self.connection.create_table(self.table_name, data)
        return ids

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document.

        Args:
            doc_id (str): document id

        """
        raise NotImplementedError("Delete not yet implemented for Faiss index.")

    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        table = self.connection.open_table(self.table_name)
        lance_query = (
            table.search(query.query_embedding)
            .limit(query.similarity_top_k)
            .nprobes(self.nprobes)
        )

        if self.refine_factor is not None:
            lance_query.refine_factor(self.refine_factor)

        results = lance_query.to_df()
        nodes = []
        for _, item in results.iterrows():
            node = Node(
                doc_id=item.id,
                text=item.text,
                relationships={
                    DocumentRelationship.SOURCE: item.doc_id,
                },
            )
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=results["score"].tolist(),
            ids=results["id"].tolist(),
        )
