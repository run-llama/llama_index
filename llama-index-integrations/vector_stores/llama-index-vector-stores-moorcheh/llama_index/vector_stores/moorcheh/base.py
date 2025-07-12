# Importing required libraries and modules
import logging
from typing import Any, List, Optional, Callable, ClassVar, Literal
import uuid
import os

# LlamaIndex internals for schema and vector store support
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.llms import LLM
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

# Moorcheh SDK for backend vector storage
from moorcheh_sdk import MoorchehClient, MoorchehError
from moorcheh_sdk import MoorchehClient

ID_KEY = "id"
VECTOR_KEY = "values"
SPARSE_VECTOR_KEY = "sparse_values"
METADATA_KEY = "metadata"

# Logger for debug/info/error output
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
        # Initialize store attributes
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

        # Fallback to env var if API key not provided
        if not self.api_key:
            self.api_key = os.getenv("MOORCHEH_API_KEY")
        if not self.api_key:
            raise ValueError("`api_key` is required for Moorcheh client initialization")

        if not self.namespace:
            raise ValueError(
                "`namespace` is required for Moorcheh client initialization"
            )

        # Initialize Moorcheh client
        print("[DEBUG] Initializing MoorchehClient")
        self._client = MoorchehClient(api_key=self.api_key)
        self.is_embedding_query = False
        self._sparse_embedding_model = sparse_embedding_model

        print("[DEBUG] Listing namespaces...")
        try:
            namespaces = self._client.list_namespaces()
            print(f"[DEBUG] Found namespaces: {namespaces}")
        except Exception as e:
            print(f"[ERROR] Failed to list namespaces: {e}")
            raise

        if self.namespace not in namespaces:
            print(f"[DEBUG] Namespace '{self.namespace}' not found. Creating...")
            try:
                self._client.create_namespace(
                    namespace_name=self.namespace,
                    type=self.namespace_type,
                    vector_dimension=self.vector_dimension,
                )
            except Exception as e:
                print(f"[ERROR] Failed to create namespace: {e}")
                raise

        print("[DEBUG] MoorchehVectorStore initialization complete.")

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

            # Add metadata if present
            if node.metadata:
                document["metadata"] = node.metadata

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
                result = self._client.upload_documents(
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

            # Add metadata, including text content
            metadata = dict(node.metadata) if node.metadata else {}
            metadata["text"] = metadata.pop(
                "text", node.get_content(metadata_mode=MetadataMode.NONE)
            )
            vector["metadata"] = metadata

            if self.add_sparse_vector and self._sparse_embedding_model is not None:
                sparse_inputs.append(node.get_content(metadata_mode=MetadataMode.EMBED))

            vectors.append(vector)

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
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i : i + self.batch_size]
            try:
                result = self._client.upload_vectors(
                    namespace_name=self.namespace, vectors=batch
                )
                logger.debug(f"Uploaded batch of {len(batch)} vectors")
            except MoorchehError as e:
                logger.error(f"Error uploading vectors batch: {e}")
                raise

        logger.info(f"Added {len(vectors)} vectors to namespace {self.namespace}")
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        try:
            if self.namespace_type == "text":
                result = self._client.delete_documents(
                    namespace_name=self.namespace, ids=[ref_doc_id]
                )
            else:
                result = self._client.delete_vectors(
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

        # Prepare search parameters
        search_kwargs = {
            "namespaces": [self.namespace],
            "top_k": query.similarity_top_k,
        }

        # Add similarity threshold if provided
        # if query.similarity_top_k is not None:
        #    search_kwargs["threshold"] = query.similarity_top_k

        # Handle query input
        if query.query_str is not None:
            search_kwargs["query"] = query.query_str
        elif query.query_embedding is not None:
            search_kwargs["query"] = query.query_embedding
        else:
            raise ValueError("Either query_str or query_embedding must be provided")

        # TODO: Add metadata filter support when available in Moorcheh SDK
        if query.filters is not None:
            logger.warning(
                "Metadata filters are not yet supported by Moorcheh integration"
            )

        try:
            # Execute search
            search_result = self._client.search(**search_kwargs)

            # Parse results
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

                # Extract text and metadata
                if self.namespace_type == "text":
                    text = result.get("text", "")
                    metadata = result.get("metadata", {})
                else:
                    # For vector namespace, text is stored in metadata
                    metadata = result.get("metadata", {})
                    text = metadata.pop("text", "")  # Remove text from metadata

                # Create node
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
            # incorporate llama_index llms
            if llm:
                vs_query = VectorStoreQuery(query_str=query, similarity_top_k=top_k)
                result = self.query(vs_query)
                context = "\n\n".join([node.text for node in result.nodes])
                prompt = f"""Use the context below to answer the question. Context:  {context} Question: {query} Answer:"""
                return llm.complete(prompt).text
            else:
                result = self._client.get_generative_answer(
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
