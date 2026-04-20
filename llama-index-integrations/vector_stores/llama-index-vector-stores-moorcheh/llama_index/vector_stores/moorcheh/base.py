import logging
import os
import uuid
import warnings
from typing import Any, Callable, ClassVar, List, Literal, Optional

from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from moorcheh_sdk import MoorchehClient, MoorchehError

from llama_index.vector_stores.moorcheh.utils import DefaultMoorchehSparseEmbedding

ID_KEY = "id"
VECTOR_KEY = "vector"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "metadata"
TEXT_KEY = "text"

logger = logging.getLogger(__name__)


class MoorchehVectorStore(BasePydanticVectorStore):
    """
    Moorcheh Vector Store.

    In this vector store, embeddings and docs are stored within a Moorcheh namespace.
    During query time, the index uses Moorcheh to query for the top k most similar nodes.

    Args:
        api_key (Optional[str]): API key for Moorcheh.
            If not provided, will look for MOORCHEH_API_KEY environment variable.
        namespace (str): Namespace name to use for this vector store.
        namespace_type (str): Type of namespace - "text" or "vector".
        vector_dimension (Optional[int]): Vector dimension for vector namespace.
        batch_size (int): Batch size for adding nodes. Defaults to DEFAULT_EMBED_BATCH_SIZE.
        **kwargs: Additional arguments to pass to MoorchehClient.

    """

    # Default values and capabilities
    DEFAULT_NAMESPACE: ClassVar[str] = "llamaindex_default"
    DEFAULT_EMBED_BATCH_SIZE: ClassVar[int] = 64  # customize as needed

    stores_text: bool = True
    flat_metadata: bool = True

    api_key: Optional[str]
    namespace: Optional[str]
    namespace_type: Optional[Literal["text", "vector"]] = None
    vector_dimension: Optional[int]
    add_sparse_vector: Optional[bool]
    ai_model: Optional[str]
    batch_size: int
    sparse_embedding_model: Optional[BaseSparseEmbedding] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        namespace_type: Optional[str] = "text",
        vector_dimension: Optional[int] = None,
        add_sparse_vector: Optional[bool] = False,
        tokenizer: Optional[Callable] = None,
        ai_model: Optional[str] = "anthropic.claude-3-7-sonnet-20250219-v1:0",
        batch_size: int = 64,
        sparse_embedding_model: Optional[BaseSparseEmbedding] = None,
    ) -> None:
        if add_sparse_vector:
            if sparse_embedding_model is not None:
                sparse_embedding_model = sparse_embedding_model
            elif tokenizer is not None:
                sparse_embedding_model = DefaultMoorchehSparseEmbedding(
                    tokenizer=tokenizer
                )
            else:
                sparse_embedding_model = DefaultMoorchehSparseEmbedding()
        else:
            sparse_embedding_model = None

        super().__init__(
            api_key=api_key,
            namespace=namespace,
            namespace_type=namespace_type,
            vector_dimension=vector_dimension,
            add_sparse_vector=add_sparse_vector,
            batch_size=batch_size,
            sparse_embedding_model=sparse_embedding_model,
            ai_model=ai_model,
        )

        if not self.api_key:
            self.api_key = os.getenv("MOORCHEH_API_KEY")
        if not self.api_key:
            raise ValueError("`api_key` is required for Moorcheh client initialization")

        if not self.namespace:
            raise ValueError(
                "`namespace` is required for Moorcheh client initialization"
            )

        logger.debug("Initializing MoorchehClient")
        self._client = MoorchehClient(api_key=self.api_key)
        # Vector namespaces should receive embedding queries from LlamaIndex.
        self.is_embedding_query = self.namespace_type == "vector"
        self._sparse_embedding_model = sparse_embedding_model
        self.namespace = namespace

        try:
            namespaces_response = self._client.namespaces.list()
            namespaces = [
                namespace["namespace_name"]
                for namespace in namespaces_response.get("namespaces", [])
            ]
        except Exception as e:
            logger.debug(f"Failed to list namespaces: {e}")
            raise

        if self.namespace in namespaces:
            logger.debug(f"Namespace '{self.namespace}' already exists.")
        else:
            logger.debug(f"Namespace '{self.namespace}' not found. Creating it.")
            try:
                self._client.namespaces.create(
                    namespace_name=self.namespace,
                    type=self.namespace_type,
                    vector_dimension=self.vector_dimension,
                )
            except Exception as e:
                logger.debug(f"Failed to create namespace: {e}")
                raise

    # _client: MoorchehClient = PrivateAttr()

    @property
    def client(self) -> MoorchehClient:
        """Return initialized Moorcheh client."""
        return self._client

    @classmethod
    def class_name(cls) -> str:
        """Return class name."""
        return "MoorchehVectorStore"

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to Moorcheh."""
        if not nodes:
            return []

        if self.namespace_type == "text":
            return self._add_text_nodes(nodes, **add_kwargs)
        else:
            return self._add_vector_nodes(nodes, **add_kwargs)

    def _to_top_level_metadata(self, metadata: Optional[dict]) -> dict:
        """Flatten metadata so Moorcheh stores it as top-level fields.

        If callers pass nested metadata as {"metadata": {...}}, unwrap it and merge.
        Explicit top-level keys win over nested keys on conflicts.
        """
        if not metadata:
            return {}

        nested_metadata = metadata.get(METADATA_KEY, {})
        if not isinstance(nested_metadata, dict):
            nested_metadata = {}

        clean_nested_metadata = {
            key: value
            for key, value in nested_metadata.items()
            if key not in {ID_KEY, TEXT_KEY, VECTOR_KEY, SPARSE_VECTOR_KEY, METADATA_KEY}
        }

        clean_top_level_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {ID_KEY, TEXT_KEY, VECTOR_KEY, SPARSE_VECTOR_KEY, METADATA_KEY}
        }
        return {**clean_nested_metadata, **clean_top_level_metadata}

    def _add_text_nodes(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Add text documents to a text namespace."""
        documents = []
        ids = []
        sparse_inputs = []

        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            ids.append(node_id)

            document = {
                "id": node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
            }

            if node.metadata:
                document.update(self._to_top_level_metadata(dict(node.metadata)))

            if self.add_sparse_vector and self._sparse_embedding_model is not None:
                sparse_inputs.append(node.get_content(metadata_mode=MetadataMode.EMBED))

            documents.append(document)

        if sparse_inputs:
            sparse_vectors = self._sparse_embedding_model.get_text_embedding_batch(
                sparse_inputs
            )
            for i, sparse_vector in enumerate(sparse_vectors):
                documents[i][SPARSE_VECTOR_KEY] = {
                    "indices": list(sparse_vector.keys()),
                    "values": list(sparse_vector.values()),
                }

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            try:
                self._client.documents.upload(
                    namespace_name=self.namespace, documents=batch
                )
                logger.debug(f"Uploaded batch of {len(batch)} documents")
            except MoorchehError as e:
                logger.error(f"Error uploading documents batch: {e}")
                raise

        logger.info(
            f"Added {len(documents)} text documents to namespace {self.namespace}"
        )
        return ids

    def _add_vector_nodes(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Add vector nodes to vector namespace."""
        vectors = []
        ids = []
        sparse_inputs = []

        if all(node.embedding is None for node in nodes):
            raise ValueError("No embeddings could be found within your nodes")
        for node in nodes:
            if node.embedding is None:
                warnings.warn(
                    f"Node {node.node_id} has no embedding for vector namespace",
                    UserWarning,
                )

            node_id = node.node_id or str(uuid.uuid4())
            ids.append(node_id)

            vector = {
                "id": node_id,
                "vector": node.embedding,
            }

            metadata = dict(node.metadata) if node.metadata else {}
            vector[TEXT_KEY] = metadata.pop(
                "text", node.get_content(metadata_mode=MetadataMode.NONE)
            )
            vector.update(self._to_top_level_metadata(metadata))

            if self.add_sparse_vector and self._sparse_embedding_model is not None:
                sparse_inputs.append(node.get_content(metadata_mode=MetadataMode.EMBED))

            vectors.append(vector)

        if sparse_inputs:
            sparse_vectors = self._sparse_embedding_model.get_text_embedding_batch(
                sparse_inputs
            )
            for i, sparse_vector in enumerate(sparse_vectors):
                vectors[i][SPARSE_VECTOR_KEY] = {
                    "indices": list(sparse_vector.keys()),
                    "values": list(sparse_vector.values()),
                }
        # Process in batches
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i : i + self.batch_size]
            try:
                self._client.vectors.upload(
                    namespace_name=self.namespace, vectors=batch
                )
                logger.debug(f"Uploaded batch of {len(batch)} vectors")
            except MoorchehError as e:
                logger.error(f"Error uploading vectors batch: {e}")
                raise

        logger.info(f"Added {len(vectors)} vectors to namespace {self.namespace}")
        return ids

    def _build_query_filter_suffix(self, query: VectorStoreQuery) -> str:
        """Translate supported LlamaIndex metadata filters to Moorcheh query tokens.

        Supports only text-query filter syntax (`#key:value`) currently.
        """
        if query.filters is None or not getattr(query.filters, "filters", None):
            return ""

        if self.namespace_type != "text":
            logger.warning(
                "Metadata filters are only translated for text namespaces in Moorcheh integration"
            )
            return ""

        filter_tokens: List[str] = []
        unsupported_found = False
        for item in query.filters.filters:
            operator = item.operator
            key = item.key
            value = item.value

            if operator == FilterOperator.EQ:
                filter_tokens.append(f"#{key}:{str(value).replace(' ', '-')}")
            elif operator == FilterOperator.IN and isinstance(value, list):
                # Moorcheh query syntax doesn't express OR groups explicitly.
                # We add all tokens; backend interprets them as query filters.
                for one_value in value:
                    filter_tokens.append(f"#{key}:{str(one_value).replace(' ', '-')}")
            else:
                unsupported_found = True

        if unsupported_found:
            logger.warning(
                "Some metadata filter operators are not supported by Moorcheh query syntax; using supported subset only"
            )

        if not filter_tokens:
            return ""

        return " " + " ".join(filter_tokens)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        try:
            if self.namespace_type == "text":
                self._client.documents.delete(
                    namespace_name=self.namespace, ids=[ref_doc_id]
                )
            else:
                self._client.vectors.delete(
                    namespace_name=self.namespace, ids=[ref_doc_id]
                )
            logger.info(
                f"Deleted document {ref_doc_id} from namespace {self.namespace}"
            )
        except MoorchehError as e:
            logger.error(f"Error deleting document {ref_doc_id}: {e}")
            raise

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query Moorcheh vector store.

        Args:
            query (VectorStoreQuery): query object

        Returns:
            VectorStoreQueryResult: query result

        """
        moorcheh_sparse_vector = None
        if (
            query.mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.HYBRID)
            and self._sparse_embedding_model is not None
        ):
            if query.query_str is None:
                raise ValueError(
                    "query_str must be specified if mode is SPARSE or HYBRID."
                )
            sparse_vector = self._sparse_embedding_model.get_query_embedding(
                query.query_str
            )
            if query.alpha is not None:
                moorcheh_sparse_vector = {
                    "indices": list(sparse_vector.keys()),
                    "values": [v * (1 - query.alpha) for v in sparse_vector.values()],
                }
            else:
                moorcheh_sparse_vector = {
                    "indices": list(sparse_vector.keys()),
                    "values": list(sparse_vector.values()),
                }
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            logger.warning(
                f"Moorcheh does not support query mode {query.mode}. "
                "Using default mode instead."
            )
        """

        search_kwargs = {
            "namespaces": [self.namespace],
            "top_k": query.similarity_top_k,
        }

        # Vector namespaces should query with embeddings, while text namespaces
        # should query with text strings. Some LlamaIndex paths may provide both.
        if self.namespace_type == "vector":
            if query.query_embedding is not None:
                search_kwargs["query"] = query.query_embedding
            elif query.query_str is not None:
                search_kwargs["query"] = query.query_str
            else:
                raise ValueError("Either query_embedding or query_str must be provided")
        else:
            if query.query_str is not None:
                search_kwargs["query"] = query.query_str
            elif query.query_embedding is not None:
                search_kwargs["query"] = query.query_embedding
            else:
                raise ValueError("Either query_str or query_embedding must be provided")

        if query.filters is not None:
            filter_suffix = self._build_query_filter_suffix(query)
            if filter_suffix and isinstance(search_kwargs.get("query"), str):
                search_kwargs["query"] = f"{search_kwargs['query']}{filter_suffix}"
        if moorcheh_sparse_vector is not None:
            search_kwargs[SPARSE_VECTOR_KEY] = moorcheh_sparse_vector

        try:
            search_result = self._client.similarity_search.query(**search_kwargs)
            nodes = []
            similarities = []
            ids = []

            results = search_result.get("results", [])
            for result in results:
                node_id = result.get("id")
                score = result.get("score", 0.0)

                if node_id is None:
                    logger.warning("Found result with no ID, skipping")
                    continue

                ids.append(node_id)
                similarities.append(score)

                if self.namespace_type == "text":
                    text = result.get(TEXT_KEY, "")
                    metadata = result.get(METADATA_KEY, {})
                else:
                    metadata = result.get(METADATA_KEY, {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    text = result.get(TEXT_KEY) or str(metadata.pop(TEXT_KEY, ""))

                node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata=metadata,
                )
                nodes.append(node)

            return VectorStoreQueryResult(
                nodes=nodes,
                similarities=similarities,
                ids=ids,
            )

        except MoorchehError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_generative_answer(
        self,
        query: str,
        top_k: int = 5,
        ai_model: str = "anthropic.claude-3-7-sonnet-20250219-v1:0",
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> str:
        """
        Get a generative AI answer using Moorcheh's built-in RAG capability.

        This method leverages Moorcheh's information-theoretic approach
        to provide context-aware answers directly from the API.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to use for context.
            **kwargs: Additional keyword arguments passed to Moorcheh.

        Returns:
            str: Generated answer string.

        """
        try:
            if llm:
                vs_query = VectorStoreQuery(query_str=query, similarity_top_k=top_k)
                result = self.query(vs_query)
                context = "\n\n".join([node.text for node in result.nodes])
                prompt = f"""Use the context below to answer the question. Context:  {context} Question: {query} Answer:"""
                return llm.complete(prompt).text
            else:
                result = self._client.answer.generate(
                    namespace=self.namespace,
                    query=query,
                    top_k=top_k,
                    ai_model=ai_model,
                    **kwargs,
                )
                return result.get("answer", "")
        except MoorchehError as e:
            logger.error(f"Error getting generative answer: {e}")
            raise


if __name__ == "__main__":
    print("MoorchehVectorStore loaded successfully.")
