"""AwaDB vector store index.

An index that is built on top of an existing vector store.

"""

import logging
import uuid
from typing import Any, List, Optional, Set

from llama_index.legacy.schema import BaseNode, MetadataMode, TextNode
from llama_index.legacy.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)


class AwaDBVectorStore(VectorStore):
    """AwaDB vector store.

    In this vector store, embeddings are stored within a AwaDB table.

    During query time, the index uses AwaDB to query for the top
    k most similar nodes.

    Args:
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance

    """

    flat_metadata: bool = True
    stores_text: bool = True
    DEFAULT_TABLE_NAME = "llamaindex_awadb"

    @property
    def client(self) -> Any:
        """Get AwaDB client."""
        return self.awadb_client

    def __init__(
        self,
        table_name: str = DEFAULT_TABLE_NAME,
        log_and_data_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with AwaDB client.
           If table_name is not specified,
           a random table name of `DEFAULT_TABLE_NAME + last segment of uuid`
           would be created automatically.

        Args:
            table_name: Name of the table created, default DEFAULT_TABLE_NAME.
            log_and_data_dir: Optional the root directory of log and data.
            kwargs: Any possible extend parameters in the future.

        Returns:
            None.
        """
        import_err_msg = "`awadb` package not found, please run `pip install awadb`"
        try:
            import awadb
        except ImportError:
            raise ImportError(import_err_msg)
        if log_and_data_dir is not None:
            self.awadb_client = awadb.Client(log_and_data_dir)
        else:
            self.awadb_client = awadb.Client()

        if table_name == self.DEFAULT_TABLE_NAME:
            table_name += "_"
            table_name += str(uuid.uuid4()).split("-")[-1]

        self.awadb_client.Create(table_name)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to AwaDB.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            Added node ids
        """
        if not self.awadb_client:
            raise ValueError("AwaDB client not initialized")

        embeddings = []
        metadatas = []
        ids = []
        texts = []
        for node in nodes:
            embeddings.append(node.get_embedding())
            metadatas.append(
                node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=self.flat_metadata
                )
            )
            ids.append(node.node_id)
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE) or "")

        self.awadb_client.AddTexts(
            "embedding_text",
            "text_embedding",
            texts,
            embeddings,
            metadatas,
            is_duplicate_texts=False,
            ids=ids,
        )

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        Returns:
            None
        """
        if len(ref_doc_id) == 0:
            return
        ids: List[str] = []
        ids.append(ref_doc_id)
        self.awadb_client.Delete(ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query : vector store query

        Returns:
            VectorStoreQueryResult: Query results
        """
        meta_filters = {}
        if query.filters is not None:
            for filter in query.filters.legacy_filters():
                meta_filters[filter.key] = filter.value

        not_include_fields: Set[str] = {"text_embedding"}
        results = self.awadb_client.Search(
            query=query.query_embedding,
            topn=query.similarity_top_k,
            meta_filter=meta_filters,
            not_include_fields=not_include_fields,
        )

        nodes = []
        similarities = []
        ids = []

        for item_detail in results[0]["ResultItems"]:
            content = ""
            meta_data = {}
            node_id = ""
            for item_key in item_detail:
                if item_key == "embedding_text":
                    content = item_detail[item_key]
                    continue
                elif item_key == "_id":
                    node_id = item_detail[item_key]
                    ids.append(node_id)
                    continue
                elif item_key == "score":
                    similarities.append(item_detail[item_key])
                    continue
                meta_data[item_key] = item_detail[item_key]

            try:
                node = metadata_dict_to_node(meta_data)
                node.set_content(content)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    meta_data
                )

                node = TextNode(
                    text=content,
                    id_=node_id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            nodes.append(node)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
