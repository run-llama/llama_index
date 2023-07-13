import logging
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores.types import (
    MetadataFilters,
    NodeWithEmbedding,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    metadata_dict_to_node,
    node_to_metadata_dict,
    legacy_metadata_dict_to_node,
)

ID_KEY = "_id"
VECTOR_KEY = "_tensor_facets"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "meta"

DEFAULT_BATCH_SIZE = 100

_logger = logging.getLogger(__name__)

class MarqoVectorStore(VectorStore):
    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(self, marqo_client: Optional[Any] = None, index_name: Optional[str] = None, url: Optional[str] = None, api_key: Optional[str] = None, text_key: str = DEFAULT_TEXT_KEY, batch_size: int = DEFAULT_BATCH_SIZE, **kwargs: Any) -> None:
        try:
            import marqo
        except ImportError:
            raise ImportError("`marqo` package not found, please run `pip install marqo`")
        self._index_name = index_name
        if marqo_client is not None:
            self._marqo_client = cast(marqo.Client, marqo_client)
        else:
            if api_key is None or url is None:
                raise ValueError("Must specify api_key and url if not directly passing in client.")
            self._marqo_client = marqo.Client(url, api_key)
        self._text_key = text_key
        self._batch_size = batch_size
        self._ensure_index()  # Ensure the index exists

    def _ensure_index(self):
        """Ensure the index exists, creating it if necessary."""
        indexes = [index.index_name for index in self._marqo_client.get_indexes()["results"]]
        if self._index_name not in indexes:
            self._marqo_client.create_index(self._index_name)

    def add(self, documents: List[Tuple[str, str]]) -> List[str]:
        entries = []
        for doc_id, doc_text in documents:
            entry = {
                ID_KEY: doc_id,  # Use the passed in ID
                self._text_key: doc_text,
                #METADATA_KEY: doc_text,  # Implement getting metadata in the future
            }
            entries.append(entry)
        response = self._marqo_client.index(self._index_name).add_documents(entries, non_tensor_fields=[METADATA_KEY])

        # response should be something like:
        # {'errors': False, 'processingTimeMs': 444.4244759997673, 'index_name': 'test', 'items': [{'_id': 'doc1', 'result': 'updated', 'status': 200}, {'_id': 'doc2', 'result': 'updated', 'status': 200}]}
        return [doc['_id'] for doc in response['items']]


    def delete(self, ref_doc_id: List[str], **delete_kwargs: Any) -> None:
        # modify so that it can accept a list of ids
        self._marqo_client.index(self._index_name).delete_documents(ref_doc_id)


    @property
    def client(self) -> Any:
        """Return Marqo client."""
        return self._marqo_client

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        response = self._marqo_client.index(self._index_name).search(
            query.query_str,
            limit=query.similarity_top_k,
            attributes_to_retrieve=["*"],
            **kwargs,
        )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for match in response["hits"]:
            try:
                node = metadata_dict_to_node(match)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                _logger.debug(
                    "Failed to parse Node metadata, fallback to legacy logic."
                )
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    match, text_key=self._text_key
                )
                text = match[self._text_key]
                id = match[ID_KEY]
                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            top_k_ids.append(match[ID_KEY])
            top_k_nodes.append(node)
            top_k_scores.append(match["_score"])

        """return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )"""
        #return (query.query_str, response)
        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
