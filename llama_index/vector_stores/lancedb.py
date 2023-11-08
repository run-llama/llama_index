"""LanceDB vector store."""
from typing import Any, List, Optional

import numpy as np
from pandas import DataFrame

from llama_index.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict


def _to_lance_filter(standard_filters: MetadataFilters) -> Any:
    """Translate standard metadata filters to Lance specific spec."""
    filters = []
    for filter in standard_filters.filters:
        if isinstance(filter.value, str):
            filters.append(filter.key + ' = "' + filter.value + '"')
        else:
            filters.append(filter.key + " = " + str(filter.value))
    return " AND ".join(filters)


def _to_llama_similarities(results: DataFrame) -> List[float]:
    keys = results.keys()
    normalized_similarities: np.ndarray
    if "score" in keys:
        normalized_similarities = np.exp(results["score"] - np.max(results["score"]))
    elif "_distance" in keys:
        normalized_similarities = np.exp(-results["_distance"])
    else:
        normalized_similarities = np.linspace(1, 0, len(results))
    return normalized_similarities.tolist()


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
    flat_metadata: bool = True

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
            import lancedb
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
        return

    def add(
        self,
        nodes: List[BaseNode],
    ) -> List[str]:
        data = []
        ids = []
        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            append_data = {
                "id": node.node_id,
                "doc_id": node.ref_doc_id,
                "vector": node.get_embedding(),
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
            }
            append_data.update(metadata)
            data.append(append_data)
            ids.append(node.node_id)

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
        table = self.connection.open_table(self.table_name)
        table.delete('document_id = "' + ref_doc_id + '"')

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for lancedb specific items that are "
                    "not supported via the generic query interface."
                )
            where = _to_lance_filter(query.filters)
        else:
            where = kwargs.pop("where", None)

        table = self.connection.open_table(self.table_name)
        lance_query = (
            table.search(query.query_embedding)
            .limit(query.similarity_top_k)
            .where(where)
            .nprobes(self.nprobes)
        )

        if self.refine_factor is not None:
            lance_query.refine_factor(self.refine_factor)

        results = lance_query.to_df()
        nodes = []
        for _, item in results.iterrows():
            node = TextNode(
                text=item.text,
                id_=item.id,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=item.doc_id),
                },
            )
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=_to_llama_similarities(results),
            ids=results["id"].tolist(),
        )
