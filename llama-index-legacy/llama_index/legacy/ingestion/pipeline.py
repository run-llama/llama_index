import asyncio
import multiprocessing
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from functools import partial, reduce
from hashlib import sha256
from itertools import repeat
from pathlib import Path
from typing import Any, Generator, List, Optional, Sequence, Union

from fsspec import AbstractFileSystem

from llama_index.legacy.bridge.pydantic import BaseModel, Field
from llama_index.legacy.embeddings.utils import resolve_embed_model
from llama_index.legacy.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from llama_index.legacy.node_parser import SentenceSplitter
from llama_index.legacy.readers.base import ReaderConfig
from llama_index.legacy.schema import (
    BaseNode,
    Document,
    MetadataMode,
    TransformComponent,
)
from llama_index.legacy.service_context import ServiceContext
from llama_index.legacy.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.legacy.storage.storage_context import DOCSTORE_FNAME
from llama_index.legacy.utils import concat_dirs
from llama_index.legacy.vector_stores.types import BasePydanticVectorStore


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


def arun_transformations_wrapper(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> List[BaseNode]:
    """Wrapper for async run_transformation. To be used in loop.run_in_executor
    within a ProcessPoolExecutor.
    """
    loop = asyncio.new_event_loop()
    nodes = loop.run_until_complete(
        arun_transformations(
            nodes=nodes,
            transformations=transformations,
            in_place=in_place,
            cache=cache,
            cache_collection=cache_collection,
            **kwargs,
        )
    )
    loop.close()
    return nodes


class DocstoreStrategy(str, Enum):
    """Document de-duplication strategy."""

    UPSERTS = "upserts"
    DUPLICATES_ONLY = "duplicates_only"
    UPSERTS_AND_DELETE = "upserts_and_delete"


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
        description="Document store to use for de-duping with a vector store.",
    )
    docstore_strategy: DocstoreStrategy = Field(
        default=DocstoreStrategy.UPSERTS, description="Document de-dup strategy."
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
        docstore_strategy: DocstoreStrategy = DocstoreStrategy.UPSERTS,
        disable_cache: bool = False,
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
            docstore_strategy=docstore_strategy,
            disable_cache=disable_cache,
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
        disable_cache: bool = False,
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
            disable_cache=disable_cache,
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
            persist_path = Path(persist_dir)
            docstore_path = str(persist_path / docstore_name)
            cache_path = str(persist_path / cache_name)

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

    def _handle_duplicates(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
        """Handle docstore duplicates by checking all hashes."""
        assert self.docstore is not None

        existing_hashes = self.docstore.get_all_document_hashes()
        current_hashes = []
        nodes_to_run = []
        for node in nodes:
            if node.hash not in existing_hashes and node.hash not in current_hashes:
                self.docstore.set_document_hash(node.id_, node.hash)
                nodes_to_run.append(node)
                current_hashes.append(node.hash)

        self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run

    def _handle_upserts(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None

        existing_doc_ids_before = set(self.docstore.get_all_document_hashes().values())
        doc_ids_from_nodes = set()
        deduped_nodes_to_run = {}
        for node in nodes:
            ref_doc_id = node.ref_doc_id if node.ref_doc_id else node.id_
            doc_ids_from_nodes.add(ref_doc_id)
            existing_hash = self.docstore.get_document_hash(ref_doc_id)
            if not existing_hash:
                # document doesn't exist, so add it
                self.docstore.set_document_hash(ref_doc_id, node.hash)
                deduped_nodes_to_run[ref_doc_id] = node
            elif existing_hash and existing_hash != node.hash:
                self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

                self.docstore.set_document_hash(ref_doc_id, node.hash)

                deduped_nodes_to_run[ref_doc_id] = node
            else:
                continue  # document exists and is unchanged, so skip it

        if self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
            # Identify missing docs and delete them from docstore and vector store
            doc_ids_to_delete = existing_doc_ids_before - doc_ids_from_nodes
            for ref_doc_id in doc_ids_to_delete:
                self.docstore.delete_document(ref_doc_id)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

        nodes_to_run = list(deduped_nodes_to_run.values())
        self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run

    @staticmethod
    def _node_batcher(
        num_batches: int, nodes: Union[List[BaseNode], List[Document]]
    ) -> Generator[Union[List[BaseNode], List[Document]], Any, Any]:
        """Yield successive n-sized chunks from lst."""
        batch_size = max(1, int(len(nodes) / num_batches))
        for i in range(0, len(nodes), batch_size):
            yield nodes[i : i + batch_size]

    def run(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        store_doc_text: bool = True,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Args:
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.
            documents (Optional[List[Document]], optional): Set of documents to be transformed. Defaults to None.
            nodes (Optional[List[BaseNode]], optional): Set of nodes to be transformed. Defaults to None.
            cache_collection (Optional[str], optional): Cache for transformations. Defaults to None.
            in_place (bool, optional): Whether transformations creates a new list for transformed nodes or modifies the
                array passed to `run_transformations`. Defaults to True.
            num_workers (Optional[int], optional): The number of parallel processes to use.
                If set to None, then sequential compute is used. Defaults to None.

        Returns:
            Sequence[BaseNode]: The set of transformed Nodes/Documents
        """
        input_nodes = self._prepare_inputs(documents, nodes)

        # check if we need to dedup
        if self.docstore is not None and self.vector_store is not None:
            if self.docstore_strategy in (
                DocstoreStrategy.UPSERTS,
                DocstoreStrategy.UPSERTS_AND_DELETE,
            ):
                nodes_to_run = self._handle_upserts(
                    input_nodes, store_doc_text=store_doc_text
                )
            elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
                nodes_to_run = self._handle_duplicates(
                    input_nodes, store_doc_text=store_doc_text
                )
            else:
                raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
        elif self.docstore is not None and self.vector_store is None:
            if self.docstore_strategy == DocstoreStrategy.UPSERTS:
                print(
                    "Docstore strategy set to upserts, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            elif self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
                print(
                    "Docstore strategy set to upserts and delete, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            nodes_to_run = self._handle_duplicates(
                input_nodes, store_doc_text=store_doc_text
            )

        else:
            nodes_to_run = input_nodes

        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )

            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                node_batches = self._node_batcher(
                    num_batches=num_workers, nodes=nodes_to_run
                )
                nodes_parallel = p.starmap(
                    run_transformations,
                    zip(
                        node_batches,
                        repeat(self.transformations),
                        repeat(in_place),
                        repeat(self.cache if not self.disable_cache else None),
                        repeat(cache_collection),
                    ),
                )
                nodes = reduce(lambda x, y: x + y, nodes_parallel, [])
        else:
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

    # ------ async methods ------

    async def _ahandle_duplicates(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
        """Handle docstore duplicates by checking all hashes."""
        assert self.docstore is not None

        existing_hashes = await self.docstore.aget_all_document_hashes()
        current_hashes = []
        nodes_to_run = []
        for node in nodes:
            if node.hash not in existing_hashes and node.hash not in current_hashes:
                await self.docstore.aset_document_hash(node.id_, node.hash)
                nodes_to_run.append(node)
                current_hashes.append(node.hash)

        await self.docstore.async_add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run

    async def _ahandle_upserts(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None

        existing_doc_ids_before = set(
            (await self.docstore.aget_all_document_hashes()).values()
        )
        doc_ids_from_nodes = set()
        deduped_nodes_to_run = {}
        for node in nodes:
            ref_doc_id = node.ref_doc_id if node.ref_doc_id else node.id_
            doc_ids_from_nodes.add(ref_doc_id)
            existing_hash = await self.docstore.aget_document_hash(ref_doc_id)
            if not existing_hash:
                # document doesn't exist, so add it
                await self.docstore.aset_document_hash(ref_doc_id, node.hash)
                deduped_nodes_to_run[ref_doc_id] = node
            elif existing_hash and existing_hash != node.hash:
                await self.docstore.adelete_ref_doc(ref_doc_id, raise_error=False)

                if self.vector_store is not None:
                    await self.vector_store.adelete(ref_doc_id)

                await self.docstore.aset_document_hash(ref_doc_id, node.hash)

                deduped_nodes_to_run[ref_doc_id] = node
            else:
                continue  # document exists and is unchanged, so skip it

        if self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
            # Identify missing docs and delete them from docstore and vector store
            doc_ids_to_delete = existing_doc_ids_before - doc_ids_from_nodes
            for ref_doc_id in doc_ids_to_delete:
                await self.docstore.adelete_document(ref_doc_id)

                if self.vector_store is not None:
                    await self.vector_store.adelete(ref_doc_id)

        nodes_to_run = list(deduped_nodes_to_run.values())
        await self.docstore.async_add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run

    async def arun(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        store_doc_text: bool = True,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        input_nodes = self._prepare_inputs(documents, nodes)

        # check if we need to dedup
        if self.docstore is not None and self.vector_store is not None:
            if self.docstore_strategy in (
                DocstoreStrategy.UPSERTS,
                DocstoreStrategy.UPSERTS_AND_DELETE,
            ):
                nodes_to_run = await self._ahandle_upserts(
                    input_nodes, store_doc_text=store_doc_text
                )
            elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
                nodes_to_run = await self._ahandle_duplicates(
                    input_nodes, store_doc_text=store_doc_text
                )
            else:
                raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
        elif self.docstore is not None and self.vector_store is None:
            if self.docstore_strategy == DocstoreStrategy.UPSERTS:
                print(
                    "Docstore strategy set to upserts, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            elif self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
                print(
                    "Docstore strategy set to upserts and delete, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            nodes_to_run = await self._ahandle_duplicates(
                input_nodes, store_doc_text=store_doc_text
            )

        else:
            nodes_to_run = input_nodes

        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )

            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=num_workers) as p:
                node_batches = self._node_batcher(
                    num_batches=num_workers, nodes=nodes_to_run
                )
                tasks = [
                    loop.run_in_executor(
                        p,
                        partial(
                            arun_transformations_wrapper,
                            transformations=self.transformations,
                            in_place=in_place,
                            cache=self.cache if not self.disable_cache else None,
                            cache_collection=cache_collection,
                        ),
                        batch,
                    )
                    for batch in node_batches
                ]
                result: List[List[BaseNode]] = await asyncio.gather(*tasks)
                nodes = reduce(lambda x, y: x + y, result, [])
        else:
            nodes = await arun_transformations(
                nodes_to_run,
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
