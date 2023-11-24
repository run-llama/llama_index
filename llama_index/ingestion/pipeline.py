import re
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional, Sequence

from fsspec import AbstractFileSystem

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from llama_index.node_parser import SentenceSplitter
from llama_index.readers.base import ReaderConfig
from llama_index.schema import BaseNode, Document, MetadataMode, TransformComponent
from llama_index.service_context import ServiceContext
from llama_index.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.storage.storage_context import DOCSTORE_FNAME
from llama_index.utils import concat_dirs
from llama_index.vector_stores.types import BasePydanticVectorStore


def remove_unstable_values(s: str) -> str:
    """Remove unstable key/value pairs.

    Examples include:
    - <__main__.Test object at 0x7fb9f3793f50>
    - <function test_fn at 0x7fb9f37a8900>
    """
    pattern = r"<[\w\s_\. ]+ at 0x[a-z0-9]+>"
    return re.sub(pattern, "", s)


def get_transformation_hash(
    nodes: List[BaseNode], transformation: TransformComponent
) -> str:
    """Get the hash of a transformation."""
    nodes_str = "".join(
        [str(node.get_content(metadata_mode=MetadataMode.ALL)) for node in nodes]
    )

    transformation_dict = transformation.to_dict()
    transform_string = remove_unstable_values(str(transformation_dict))

    return sha256((nodes_str + transform_string).encode("utf-8")).hexdigest()


def run_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> List[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        if cache is not None:
            hash = get_transformation_hash(nodes, transform)
            cached_nodes = cache.get(hash, collection=cache_collection)
            if cached_nodes is not None:
                nodes = cached_nodes
            else:
                nodes = transform(nodes, **kwargs)
                cache.put(hash, nodes, collection=cache_collection)
        else:
            nodes = transform(nodes, **kwargs)

    return nodes


async def arun_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> List[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        if cache is not None:
            hash = get_transformation_hash(nodes, transform)

            cached_nodes = cache.get(hash, collection=cache_collection)
            if cached_nodes is not None:
                nodes = cached_nodes
            else:
                nodes = await transform.acall(nodes, **kwargs)
                cache.put(hash, nodes, collection=cache_collection)
        else:
            nodes = await transform.acall(nodes, **kwargs)

    return nodes


class IngestionPipeline(BaseModel):
    """An ingestion pipeline that can be applied to data."""

    transformations: List[TransformComponent] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    reader: Optional[ReaderConfig] = Field(description="Reader to use to read the data")
    vector_store: Optional[BasePydanticVectorStore] = Field(
        description="Vector store to use to store the data"
    )
    cache: IngestionCache = Field(
        default_factory=IngestionCache,
        description="Cache to use to store the data",
    )
    docstore: Optional[BaseDocumentStore] = Field(
        default=None,
        description="Document store to use to store the data for de-duping",
    )
    dedup_key: Optional[str] = Field(
        default=None,
        description="Metadata key to use for de-duping documents. If not specified, will use the ref_doc_id.",
    )
    disable_cache: bool = Field(default=False, description="Disable the cache")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        transformations: Optional[List[TransformComponent]] = None,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        cache: Optional[IngestionCache] = None,
        docstore: Optional[BaseDocumentStore] = None,
        dedup_key: Optional[str] = None,
    ) -> None:
        if transformations is None:
            transformations = self._get_default_transformations()

        super().__init__(
            transformations=transformations,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
            cache=cache or IngestionCache(),
            docstore=docstore,
            dedup_key=dedup_key,
        )

    @classmethod
    def from_service_context(
        cls,
        service_context: ServiceContext,
        reader: Optional[ReaderConfig] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        cache: Optional[IngestionCache] = None,
        docstore: Optional[BaseDocumentStore] = None,
        dedup_key: Optional[str] = None,
    ) -> "IngestionPipeline":
        transformations = [
            *service_context.transformations,
            service_context.embed_model,
        ]

        return cls(
            transformations=transformations,
            reader=reader,
            documents=documents,
            vector_store=vector_store,
            cache=cache,
            docstore=docstore,
            dedup_key=dedup_key,
        )

    def persist(
        self,
        persist_dir: str = "./pipeline_storage",
        fs: Optional[AbstractFileSystem] = None,
        cache_name: str = DEFAULT_CACHE_NAME,
        docstore_name: str = DOCSTORE_FNAME,
    ) -> None:
        """Persist the pipeline to disk."""
        if fs is not None:
            persist_dir = str(persist_dir)  # NOTE: doesn't support Windows here
            docstore_path = concat_dirs(persist_dir, docstore_name)
            cache_path = concat_dirs(persist_dir, cache_name)

        else:
            persist_dir = Path(persist_dir)
            docstore_path = str(persist_dir / docstore_name)
            cache_path = str(persist_dir / cache_name)

        self.cache.persist(cache_path, fs=fs)
        if self.docstore is not None:
            self.docstore.persist(docstore_path, fs=fs)

    def load(
        self,
        persist_dir: str = "./pipeline_storage",
        fs: Optional[AbstractFileSystem] = None,
        cache_name: str = DEFAULT_CACHE_NAME,
        docstore_name: str = DOCSTORE_FNAME,
    ) -> None:
        """Load the pipeline from disk."""
        if fs is not None:
            self.cache = IngestionCache.from_persist_path(
                concat_dirs(persist_dir, cache_name), fs=fs
            )
            self.docstore = SimpleDocumentStore.from_persist_path(
                concat_dirs(persist_dir, docstore_name), fs=fs
            )
        else:
            self.cache = IngestionCache.from_persist_path(
                str(Path(persist_dir) / cache_name)
            )
            self.docstore = SimpleDocumentStore.from_persist_path(
                str(Path(persist_dir) / docstore_name)
            )

    def _get_default_transformations(self) -> List[TransformComponent]:
        return [
            SentenceSplitter(),
            resolve_embed_model("default"),
        ]

    def _prepare_inputs(
        self, documents: Optional[List[Document]], nodes: Optional[List[BaseNode]]
    ) -> List[Document]:
        input_nodes: List[BaseNode] = []
        if documents is not None:
            input_nodes += documents

        if nodes is not None:
            input_nodes += nodes

        if self.documents is not None:
            input_nodes += self.documents

        if self.reader is not None:
            input_nodes += self.reader.read()

        return input_nodes

    def run(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        input_nodes = self._prepare_inputs(documents, nodes)

        # check if we need to de-
        nodes_to_run = []
        if self.docstore is not None:
            for node in input_nodes:
                if self.dedup_key is not None:
                    doc_key = str(node.metadata.get(self.dedup_key, None))
                else:
                    doc_key = node.ref_doc_id or node.id_

                if doc_key and not self.docstore.document_exists(doc_key):
                    # document doesn't exist, so add it
                    self.docstore.add_documents(
                        [node], allow_update=False, ref_doc_key=doc_key
                    )
                    self.docstore.set_document_hash(doc_key, node.hash)
                    nodes_to_run.append(node)
                elif doc_key and self.docstore.document_exists(doc_key):
                    existing_hash = self.docstore.get_document_hash(doc_key)

                    # update
                    if existing_hash != node.hash:
                        self.docstore.delete_ref_doc(doc_key, raise_error=False)

                        if self.vector_store is not None:
                            self.vector_store.delete(doc_key)

                        self.docstore.add_documents([node], ref_doc_key=doc_key)
                        self.docstore.set_document_hash(doc_key, node.hash)

                        nodes_to_run.append(node)
        else:
            nodes_to_run = input_nodes

        nodes = run_transformations(
            nodes_to_run,
            self.transformations,
            show_progress=show_progress,
            cache=self.cache if not self.disable_cache else None,
            cache_collection=cache_collection,
            in_place=in_place,
            **kwargs,
        )

        if self.vector_store is not None:
            self.vector_store.add([n for n in nodes if n.embedding is not None])

        return nodes

    async def arun(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        input_nodes = self._prepare_inputs(documents, nodes)

        nodes = await arun_transformations(
            input_nodes,
            self.transformations,
            show_progress=show_progress,
            cache=self.cache if not self.disable_cache else None,
            cache_collection=cache_collection,
            in_place=in_place,
            **kwargs,
        )

        if self.vector_store is not None:
            await self.vector_store.async_add(
                [n for n in nodes if n.embedding is not None]
            )

        return nodes
