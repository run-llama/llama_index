"""LanceDB vector store."""
from typing import Any, List, Optional

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class LanceDBVectorStore(VectorStore):
    """The LanceDB Vector Store.

    Stores text and embeddings in LanceDB. The vector store will open an existing
        LanceDB dataset or create the dataset if it does not exist.

    Args:
        uri (str, required): Location where LanceDB will store its files.
        table_name (str, optional): The table name where the embeddings will be stored.
            Defaults to "vectors".
        nprobes (int, optional): The number of probes used.
            A higher number makes search more accurate but also slower.
            Defaults to 20.
        refine_factor: (int, optional): Refine the results by reading extra elements
            and re-ranking them in memory.
            Defaults to None

    Raises:
        ImportError: Unable to import `lancedb`.

    Returns:
        LanceDBVectorStore: VectorStore that supports creating LanceDB datasets and
            querying it.
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

    @property
    def client(self) -> None:
        """Get client."""
        return None

    def add(
        self,
        embedding_results: List[NodeWithEmbedding],
    ) -> List[str]:
        data = []
        ids = []
        for result in embedding_results:
            data.append(
                {
                    "id": result.id,
                    "doc_id": result.ref_doc_id,
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

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        raise NotImplementedError("Delete not yet implemented for LanceDB.")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if query.filters is not None:
            raise ValueError("Metadata filters not implemented for LanceDB yet.")

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
