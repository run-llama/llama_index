from llama_index.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.bridge.pydantic import BaseModel, Field, PrivateAttr # type: ignore
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import BaseNode, RelatedNodeInfo, TextNode
import logging
from typing import Any, cast, Dict, List, Optional, Sequence
import uuid


_logger = logging.getLogger(__name__)
_import_err_msg = (
    "`google.generativeai` package not found, please run `pip install google-generativeai`"
)
_default_doc_id = 'default-doc'


google_service_context = ServiceContext.from_defaults(
    # Avoids instantiating OpenAI as the default model.
    llm=None,
    # Avoids instantiating HuggingFace as the default model.
    embed_model=None)


class GoogleVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    is_embedding_query: bool = False

    # This is not the Google's corpus name but an ID generated in the LlamaIndex
    # world.
    corpus_id: str = Field(frozen=True)

    _client: Any = PrivateAttr()

    def __init__(self, *, client: Any, **kwargs: Any):
        try:
            import google.ai.generativelanguage as genai
        except ImportError:
            raise ImportError(_import_err_msg)

        super().__init__(**kwargs)

        assert isinstance(client, genai.RetrieverServiceClient)
        self._client = client

    @classmethod
    def from_corpus(cls, *, corpus_id: str) -> 'GoogleVectorStore':
        try:
            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(
            f"\n\nGoogleVectorStore.from_corpus(corpus_id={corpus_id})")

        return cls(
            corpus_id=corpus_id,
            client=genaix.build_retriever()
        )

    @classmethod
    def create_corpus(
        cls, *, corpus_id: Optional[str] = None, display_name: Optional[str] = None
    ) -> 'GoogleVectorStore':
        try:
            import google.ai.generativelanguage as genai
            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(
            f"\n\nGoogleVectorStore.create_corpus(new_corpus_id={corpus_id}, new_display_name={display_name})")

        client = genaix.build_retriever()
        new_corpus_id = corpus_id or str(uuid.uuid4())
        new_corpus = genaix.create_corpus(
            corpus_id=new_corpus_id, display_name=display_name, client=client)
        name = genaix.EntityName.from_str(new_corpus.name)
        return cls(corpus_id=name.corpus_id, client=client)

    @classmethod
    def class_name(cls) -> str:
        return "GoogleVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: List[BaseNode]) -> List[str]:
        try:
            import google.ai.generativelanguage as genai
            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(f"\n\nGoogleVectorStore.add(nodes={nodes})")

        client = cast(genai.RetrieverServiceClient, self.client)

        for nodeGroup in _groupNodesBySource(nodes):
            source = nodeGroup.source_node
            document_id = source.node_id
            document = genaix.get_document(
                corpus_id=self.corpus_id, document_id=document_id, client=client)

            if not document:
                genaix.create_document(
                    corpus_id=self.corpus_id,
                    display_name=source.metadata.get("file_name", None),
                    document_id=document_id,
                    metadata=source.metadata,
                    client=client,
                )

            genaix.batch_create_chunk(
                corpus_id=self.corpus_id,
                document_id=document_id,
                texts=[node.get_content() for node in nodeGroup.nodes],
                metadatas=[node.metadata for node in nodeGroup.nodes],
                client=client,
            )

        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        try:
            import google.ai.generativelanguage as genai
            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(f"\n\nGoogleVectorStore.delete(ref_doc_id={ref_doc_id})")

        client = cast(genai.RetrieverServiceClient, self.client)
        genaix.delete_document(corpus_id=self.corpus_id,
                               document_id=ref_doc_id, client=client)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        try:
            import google.ai.generativelanguage as genai
            import llama_index.vector_stores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        _logger.debug(f"\n\nGoogleVectorStore.query(query={query})")

        client = cast(genai.RetrieverServiceClient, self.client)

        relevant_chunks: List[genai.RelevantChunk] = []
        if query.doc_ids is None:
            relevant_chunks = genaix.query_corpus(
                corpus_id=self.corpus_id,
                query=query.query_str or "what?",
                filter=_convertFilter(query.filters),
                k=query.similarity_top_k,
                client=client,
            )
        else:
            for doc_id in query.doc_ids:
                relevant_chunks.extend(
                    genaix.query_document(
                        corpus_id=self.corpus_id,
                        document_id=doc_id,
                        query=query.query_str or "what?",
                        filter=_convertFilter(query.filters),
                        k=query.similarity_top_k,
                        client=client,
                    )
                )

        return VectorStoreQueryResult(
            nodes=[
                TextNode(text=chunk.chunk.data.string_value,
                         id_=_extract_chunk_id(chunk.chunk.name))
                for chunk in relevant_chunks
            ],
            ids=[_extract_chunk_id(chunk.chunk.name)
                 for chunk in relevant_chunks],
            similarities=[
                chunk.chunk_relevance_score
                for chunk in relevant_chunks
            ])


def _extract_chunk_id(entity_name: str) -> str:
    try:
        import llama_index.vector_stores.google.generativeai.genai_extension as genaix
    except ImportError:
        raise ImportError(_import_err_msg)

    id = genaix.EntityName.from_str(entity_name).chunk_id
    assert id is not None
    return id


class _NodeGroup(BaseModel):
    """Every node in nodes have the same source node."""
    source_node: RelatedNodeInfo
    nodes: List[BaseNode]


def _groupNodesBySource(nodes: Sequence[BaseNode]) -> List[_NodeGroup]:
    """Returns a list of lists of nodes where each list has all the nodes
    from the same document."""
    groups: Dict[str, _NodeGroup] = {}
    for node in nodes:
        source_node: RelatedNodeInfo
        if isinstance(node.source_node, RelatedNodeInfo):
            source_node = node.source_node
        else:
            source_node = RelatedNodeInfo(node_id=_default_doc_id)

        if source_node.node_id not in groups:
            groups[source_node.node_id] = _NodeGroup(
                source_node=source_node, nodes=[])

        groups[source_node.node_id].nodes.append(node)

    return list(groups.values())


def _convertFilter(fs: Optional[MetadataFilters]) -> Dict[str, Any]:
    if fs == None:
        return {}
    assert isinstance(fs, MetadataFilters)
    return {
        f.key: f.value
        for f in fs.filters
    }
