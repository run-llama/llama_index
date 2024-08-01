"""Multidoc Autoretriever."""

from typing import Any, Dict, List, Optional, cast

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    BaseRetriever,
    RecursiveRetriever,
    VectorIndexAutoRetriever,
)
from llama_index.core.schema import BaseNode, Document, IndexNode, NodeWithScore
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreInfo,
)
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.weaviate import WeaviateVectorStore


class IndexAutoRetriever(BaseRetriever):
    """Index auto-retriever.

    Simple wrapper around VectorIndexAutoRetriever to convert
    text nodes to index nodes.

    """

    def __init__(self, retriever: VectorIndexAutoRetriever):
        """Init params."""
        self.retriever = retriever

    def _retrieve(self, query_bundle: QueryBundle):
        """Convert nodes to index node."""
        retrieved_nodes = self.retriever.retrieve(query_bundle)
        new_retrieved_nodes = []
        for retrieved_node in retrieved_nodes:
            index_id = retrieved_node.metadata["index_id"]
            index_node = IndexNode.from_text_node(
                retrieved_node.node, index_id=index_id
            )
            new_retrieved_nodes.append(
                NodeWithScore(node=index_node, score=retrieved_node.score)
            )
        return new_retrieved_nodes


class MultiDocAutoRetrieverPack(BaseLlamaPack):
    """Multi-doc auto-retriever pack.

    Uses weaviate as the underlying storage.

    Args:
        docs (List[Document]): A list of documents to index.
        **kwargs: Keyword arguments to pass to the underlying index.

    """

    def __init__(
        self,
        weaviate_client: Any,
        doc_metadata_index_name: str,
        doc_chunks_index_name: str,
        metadata_nodes: List[BaseNode],
        docs: List[Document],
        doc_metadata_schema: VectorStoreInfo,
        auto_retriever_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        import weaviate

        # do some validation
        if len(docs) != len(metadata_nodes):
            raise ValueError(
                "The number of metadata nodes must match the number of documents."
            )

        # authenticate
        client = cast(weaviate.Client, weaviate_client)
        # auth_config = weaviate.AuthApiKey(api_key="")
        # client = weaviate.Client(
        #     "https://<weaviate-cluster>.weaviate.network",
        #     auth_client_secret=auth_config,
        # )

        # initialize two vector store classes corresponding to the two index names
        metadata_store = WeaviateVectorStore(
            weaviate_client=client, index_name=doc_metadata_index_name
        )
        metadata_sc = StorageContext.from_defaults(vector_store=metadata_store)
        # index VectorStoreIndex
        # Since "new_docs" are concise summaries, we can directly feed them as nodes into VectorStoreIndex
        index = VectorStoreIndex(metadata_nodes, storage_context=metadata_sc)
        if verbose:
            print("Indexed metadata nodes.")

        # construct separate Weaviate Index with original docs. Define a separate query engine with query engine mapping to each doc id.
        chunks_store = WeaviateVectorStore(
            weaviate_client=client, index_name=doc_chunks_index_name
        )
        chunks_sc = StorageContext.from_defaults(vector_store=chunks_store)
        doc_index = VectorStoreIndex.from_documents(docs, storage_context=chunks_sc)
        if verbose:
            print("Indexed source document nodes.")

        # setup auto retriever
        auto_retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=doc_metadata_schema,
            **(auto_retriever_kwargs or {}),
        )
        self.index_auto_retriever = IndexAutoRetriever(retriever=auto_retriever)
        if verbose:
            print("Setup autoretriever over metadata.")

        # define per-document retriever
        self.retriever_dict = {}
        for doc in docs:
            index_id = doc.metadata["index_id"]
            # filter for the specific doc id
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="index_id", operator=FilterOperator.EQ, value=index_id
                    ),
                ]
            )
            retriever = doc_index.as_retriever(filters=filters)

            self.retriever_dict[index_id] = retriever

        if verbose:
            print("Setup per-document retriever.")

        # setup recursive retriever
        self.recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": self.index_auto_retriever, **self.retriever_dict},
            verbose=True,
        )
        if verbose:
            print("Setup recursive retriever.")

        # plug into query engine
        llm = OpenAI(model="gpt-3.5-turbo")
        self.query_engine = RetrieverQueryEngine.from_args(
            self.recursive_retriever, llm=llm
        )

    def get_modules(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing the internals of the LlamaPack.

        Returns:
            Dict[str, Any]: A dictionary containing the internals of the
            LlamaPack.
        """
        return {
            "index_auto_retriever": self.index_auto_retriever,
            "retriever_dict": self.retriever_dict,
            "recursive_retriever": self.recursive_retriever,
            "query_engine": self.query_engine,
        }

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs queries against the index.

        Returns:
            Any: A response from the query engine.
        """
        return self.query_engine.query(*args, **kwargs)
