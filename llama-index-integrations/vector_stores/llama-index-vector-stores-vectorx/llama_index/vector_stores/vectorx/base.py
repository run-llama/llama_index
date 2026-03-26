import logging
from collections import Counter
import json
from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)

ID_KEY = "id"
VECTOR_KEY = "values"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "metadata"

DEFAULT_BATCH_SIZE = 100

_logger = logging.getLogger(__name__)

from llama_index.core.vector_stores.types import FilterOperator

reverse_operator_map = {
    FilterOperator.EQ: "$eq",
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
}


def build_dict(input_batch: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Build a list of sparse dictionaries from a batch of input_ids.

    NOTE: taken from https://www.pinecone.io/learn/hybrid-search-intro/.

    """
    # store a batch of sparse embeddings
    sparse_emb = []
    # iterate through input batch
    for token_ids in input_batch:
        indices = []
        values = []
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        for idx in d:
            indices.append(idx)
            values.append(float(d[idx]))
        sparse_emb.append({"indices": indices, "values": values})
    # return sparse_emb list
    return sparse_emb


def generate_sparse_vectors(
    context_batch: List[str], tokenizer: Callable
) -> List[Dict[str, Any]]:
    """
    Generate sparse vectors from a batch of contexts.

    NOTE: taken from https://www.pinecone.io/learn/hybrid-search-intro/.

    """
    # create batch of input_ids
    outputs = tokenizer(context_batch)

    if not isinstance(outputs, dict) or "input_ids" not in outputs:
        raise ValueError("Tokenizer must return a dict with 'input_ids'.")

    input_ids = outputs["input_ids"]
    # create sparse dictionaries
    return build_dict(input_ids)


class VectorXVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = False

    api_token: Optional[str]
    encryption_key: Optional[str]
    index_name: Optional[str]
    space_type: Optional[str]
    dimension: Optional[int]
    insert_kwargs: Optional[Dict]
    add_sparse_vector: bool
    text_key: str
    batch_size: int
    remove_text_from_metadata: bool

    _vectorx_index: Any = PrivateAttr()

    def __init__(
        self,
        vectorx_index: Optional[Any] = None,
        api_token: Optional[str] = None,
        encryption_key: Optional[str] = None,
        index_name: Optional[str] = None,
        space_type: Optional[str] = "cosine",
        dimension: Optional[int] = None,
        insert_kwargs: Optional[Dict] = None,
        add_sparse_vector: bool = False,
        text_key: str = DEFAULT_TEXT_KEY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        remove_text_from_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        insert_kwargs = insert_kwargs or {}

        super().__init__(
            index_name=index_name,
            api_token=api_token,
            encryption_key=encryption_key,
            space_type=space_type,
            dimension=dimension,
            insert_kwargs=insert_kwargs,
            add_sparse_vector=add_sparse_vector,
            text_key=text_key,
            batch_size=batch_size,
            remove_text_from_metadata=remove_text_from_metadata,
        )

        # Use existing vectorx_index or initialize a new one
        self._vectorx_index = vectorx_index or self._initialize_vectorx_index(
            api_token, encryption_key, index_name, dimension, space_type
        )

    @classmethod
    def _initialize_vectorx_index(
        cls,
        api_token: Optional[str],
        encryption_key: Optional[str],
        index_name: Optional[str],
        dimension: Optional[int] = None,
        space_type: Optional[str] = "cosine",
    ) -> Any:
        """Initialize VectorX index using the current API."""
        try:
            from vecx.vectorx import VectorX
        except ImportError as e:
            raise ImportError(
                "Could not import `vecx` package. "
                "Please install it with `pip install vecx`."
            ) from e

        # Initialize VectorX client
        vx = VectorX(token=api_token)

        try:
            # Try to get existing index
            index = vx.get_index(name=index_name, key=encryption_key)
            _logger.info(f"Retrieved existing index: {index_name}")
            return index
        except Exception as e:
            if dimension is None:
                raise ValueError(
                    "Must provide dimension when creating a new index"
                ) from e

            # Create a new index if it doesn't exist
            _logger.info(f"Creating new index: {index_name}")
            vx.create_index(
                name=index_name,
                dimension=dimension,
                key=encryption_key,
                space_type=space_type,
            )
            return vx.get_index(name=index_name, key=encryption_key)

    @classmethod
    def from_params(
        cls,
        api_token: Optional[str] = None,
        encryption_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        space_type: str = "cosine",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "VectorXVectorStore":
        """Create VectorXVectorStore from parameters."""
        vectorx_index = cls._initialize_vectorx_index(
            api_token, encryption_key, index_name, dimension, space_type
        )

        return cls(
            vectorx_index=vectorx_index,
            api_token=api_token,
            encryption_key=encryption_key,
            index_name=index_name,
            dimension=dimension,
            space_type=space_type,
            batch_size=batch_size,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VectorXVectorStore"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        ids = []
        entries = []

        for node in nodes:
            node_id = node.node_id
            metadata = node_to_metadata_dict(node)

            # Filter values must be simple key-value pairs
            filter_data = {}
            if "file_name" in metadata:
                filter_data["file_name"] = metadata["file_name"]
            if "doc_id" in metadata:
                filter_data["doc_id"] = metadata["doc_id"]
            if "category" in metadata:
                filter_data["category"] = metadata["category"]
            if "difficulty" in metadata:
                filter_data["difficulty"] = metadata["difficulty"]
            if "language" in metadata:
                filter_data["language"] = metadata["language"]
            if "field" in metadata:
                filter_data["field"] = metadata["field"]
            if "type" in metadata:
                filter_data["type"] = metadata["type"]
            if "feature" in metadata:
                filter_data["feature"] = metadata["feature"]

            entry = {
                "id": node_id,
                "vector": node.get_embedding(),
                "meta": metadata,
                "filter": filter_data,
            }

            ids.append(node_id)
            entries.append(entry)

        # Batch insert to avoid hitting API limits
        batch_size = self.batch_size
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            self._vectorx_index.upsert(batch)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The id of the document to delete.

        """
        try:
            self._vectorx_index.delete_with_filter({"doc_id": ref_doc_id})
        except Exception as e:
            _logger.error(f"Error deleting vectors for doc_id {ref_doc_id}: {e}")

    @property
    def client(self) -> Any:
        """Return vectorX index client."""
        return self._vectorx_index

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object containing query parameters

        """
        if not hasattr(self._vectorx_index, "dimension"):
            # Get dimension from index if available, otherwise try to infer from query
            try:
                dimension = self._vectorx_index.describe()["dimension"]
            except Exception:
                if query.query_embedding is not None:
                    dimension = len(query.query_embedding)
                else:
                    raise ValueError("Could not determine vector dimension")
        else:
            dimension = self._vectorx_index.dimension

        query_embedding = [0.0] * dimension  # Default empty vector
        filters = {}

        # Apply any metadata filters if provided
        if query.filters is not None:
            for filter_item in query.filters.filters:
                # Case 1: MetadataFilter object
                if (
                    hasattr(filter_item, "key")
                    and hasattr(filter_item, "value")
                    and hasattr(filter_item, "operator")
                ):
                    op_symbol = reverse_operator_map.get(filter_item.operator)
                    if not op_symbol:
                        raise ValueError(
                            f"Unsupported filter operator: {filter_item.operator}"
                        )

                    if filter_item.key not in filters:
                        filters[filter_item.key] = {}

                    filters[filter_item.key][op_symbol] = filter_item.value

                # Case 2: Raw dict, e.g. {"category": {"$eq": "programming"}}
                elif isinstance(filter_item, dict):
                    for key, op_dict in filter_item.items():
                        if isinstance(op_dict, dict):
                            for op, val in op_dict.items():
                                if key not in filters:
                                    filters[key] = {}
                                filters[key][op] = val
                else:
                    raise ValueError(f"Unsupported filter format: {filter_item}")

        _logger.info(f"Final structured filters: {filters}")

        # Use the query embedding if provided
        if query.query_embedding is not None:
            query_embedding = cast(List[float], query.query_embedding)
            if query.alpha is not None and query.mode == VectorStoreQueryMode.HYBRID:
                # Apply alpha scaling in hybrid mode
                query_embedding = [v * query.alpha for v in query_embedding]

        # Execute query
        try:
            results = self._vectorx_index.query(
                vector=query_embedding,
                top_k=query.similarity_top_k,
                filter=filters if filters else None,
                include_vectors=True,
            )
        except Exception as e:
            _logger.error(f"Error querying VectorX: {e}")
            raise

        # Process results
        nodes = []
        similarities = []
        ids = []

        for result in results:
            node_id = result["id"]
            score = result["similarity"]

            # Get metadata from result
            metadata = result.get("meta", {})

            # Create node from metadata
            if self.flat_metadata:
                node = metadata_dict_to_node(
                    metadata=metadata,
                    text=metadata.pop(self.text_key, None),
                    id_=node_id,
                )
            else:
                metadata_dict, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata=metadata,
                    text_key=self.text_key,
                )

                # Create TextNode with the extracted metadata
                # Step 1: Get the JSON string from "_node_content"
                _node_content_str = metadata.get("_node_content", "{}")

                # Step 2: Convert JSON string to Python dict
                try:
                    node_content = json.loads(_node_content_str)
                except json.JSONDecodeError:
                    node_content = {}

                # Step 3: Get the text
                text = node_content.get(self.text_key, "")
                node = TextNode(
                    text=text,
                    metadata=metadata_dict,
                    relationships=relationships,
                    node_id=node_id,
                )

                # Add any node_info properties to the node
                for key, val in node_info.items():
                    if hasattr(node, key):
                        setattr(node, key, val)

            # If embedding was returned in the results, add it to the node
            if "vector" in result:
                node.embedding = result["vector"]

            nodes.append(node)
            similarities.append(score)
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
