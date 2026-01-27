import logging

import ray
import asyncio
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy

from llama_index.ingestion.ray.transform import RayTransformComponent
from llama_index.ingestion.ray.utils import ray_serialize_node, ray_deserialize_node

from typing import Any, List, Optional, Sequence
from llama_index.core.ingestion.cache import IngestionCache


from llama_index.core.constants import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PROJECT_NAME,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.readers.base import ReaderConfig
from llama_index.core.schema import (
    BaseNode,
    Document,
)
from llama_index.core.storage.docstore import (
    BaseDocumentStore,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore

dispatcher = get_dispatcher(__name__)
logger = logging.getLogger(__name__)


def run_transformations(
    nodes: Sequence[BaseNode],
    transformations: Sequence[RayTransformComponent],
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """
    Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.

    """
    ds = ray.data.from_items([ray_serialize_node(node) for node in nodes])

    for transform in transformations:
        ds = transform(ds, **kwargs)

    return [ray_deserialize_node(serialized_node) for serialized_node in ds.take_all()]


async def arun_transformations(
    nodes: Sequence[BaseNode],
    transformations: Sequence[RayTransformComponent],
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """
    Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.

    """
    ds = ray.data.from_items([ray_serialize_node(node) for node in nodes])

    for transform in transformations:
        ds = transform(ds, **kwargs)

    rows = await asyncio.to_thread(ds.take_all)
    return [ray_deserialize_node(serialized_node) for serialized_node in rows]


class RayIngestionPipeline(IngestionPipeline):
    """
    An ingestion pipeline that can be applied to data using a Ray cluster.

    Args:
        name (str, optional):
            Unique name of the ingestion pipeline. Defaults to DEFAULT_PIPELINE_NAME.
        project_name (str, optional):
            Unique name of the project. Defaults to DEFAULT_PROJECT_NAME.
        transformations (List[RayTransformComponent], optional):
            Ray transformations to apply to the data. Defaults to None.
        documents (Optional[Sequence[Document]], optional):
            Documents to ingest. Defaults to None.
        readers (Optional[List[ReaderConfig]], optional):
            Reader to use to read the data. Defaults to None.
        vector_store (Optional[BasePydanticVectorStore], optional):
            Vector store to use to store the data. Defaults to None.
        docstore (Optional[BaseDocumentStore], optional):
            Document store to use for de-duping with a vector store. Defaults to None.
        docstore_strategy (DocstoreStrategy, optional):
            Document de-dup strategy. Defaults to DocstoreStrategy.UPSERTS.

    Examples:
        ```python
        import ray
        from llama_index.core import Document
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core.extractors import TitleExtractor
        from llama_index.ingestion.ray import RayIngestionPipeline, RayTransformComponent

        # Start a new cluster (or connect to an existing one)
        ray.init()

        # Create transformations
        transformations=[
            RayTransformComponent(
                transform_class=TitleExtractor,
                map_batches_kwargs={
                    "batch_size": 10,  # Define the batch size
                },
            ),
            RayTransformComponent(
                transform_class=OpenAIEmbedding,
                map_batches_kwargs={
                    "batch_size": 10,
                },
            ),
        ]

        # Create the Ray ingestion pipeline
        pipeline = RayIngestionPipeline(
            transformations=transformations
        )

        # Run the pipeline with many documents
        nodes = pipeline.run(documents=[Document.example()] * 100)
        ```

    """

    transformations: List[RayTransformComponent] = Field(
        description="Transformations to apply to the data with Ray"
    )

    def __init__(
        self,
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        transformations: Optional[List[RayTransformComponent]] = None,
        readers: Optional[List[ReaderConfig]] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        docstore: Optional[BaseDocumentStore] = None,
        docstore_strategy: DocstoreStrategy = DocstoreStrategy.UPSERTS,
    ) -> None:
        BaseModel.__init__(
            self,
            name=name,
            project_name=project_name,
            transformations=transformations,
            readers=readers,
            documents=documents,
            vector_store=vector_store,
            cache=IngestionCache(),
            docstore=docstore,
            docstore_strategy=docstore_strategy,
            disable_cache=True,  # Caching is disabled as Ray processes transformations lazily
        )

    @dispatcher.span
    def run(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[Sequence[BaseNode]] = None,
        store_doc_text: bool = True,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Run a series of transformations on a set of nodes.

        If a vector store is provided, nodes with embeddings will be added to the vector store.

        If a vector store + docstore are provided, the docstore will be used to de-duplicate documents.

        Args:
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.
            documents (Optional[List[Document]], optional): Set of documents to be transformed. Defaults to None.
            nodes (Optional[Sequence[BaseNode]], optional): Set of nodes to be transformed. Defaults to None.
            store_doc_text (bool, optional): Whether to store the document texts. Defaults to True.

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
                nodes_to_run = self._handle_upserts(input_nodes)
            elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
                nodes_to_run = self._handle_duplicates(input_nodes)
            else:
                raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
        elif self.docstore is not None and self.vector_store is None:
            if self.docstore_strategy == DocstoreStrategy.UPSERTS:
                logger.info(
                    "Docstore strategy set to upserts, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            elif self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
                logger.info(
                    "Docstore strategy set to upserts and delete, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            nodes_to_run = self._handle_duplicates(input_nodes)
        else:
            nodes_to_run = input_nodes

        nodes = run_transformations(
            nodes_to_run,
            self.transformations,
            show_progress=show_progress,
            **kwargs,
        )

        if self.vector_store is not None:
            nodes_with_embeddings = [n for n in nodes if n.embedding is not None]
            if nodes_with_embeddings:
                self.vector_store.add(nodes_with_embeddings)

        if self.docstore is not None:
            self._update_docstore(nodes_to_run, store_doc_text=store_doc_text)

        return nodes

    @dispatcher.span
    async def arun(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[Sequence[BaseNode]] = None,
        store_doc_text: bool = True,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Run a series of transformations on a set of nodes.

        If a vector store is provided, nodes with embeddings will be added to the vector store.

        If a vector store + docstore are provided, the docstore will be used to de-duplicate documents.

        Args:
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.
            documents (Optional[List[Document]], optional): Set of documents to be transformed. Defaults to None.
            nodes (Optional[Sequence[BaseNode]], optional): Set of nodes to be transformed. Defaults to None.
            store_doc_text (bool, optional): Whether to store the document texts. Defaults to True.

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
                logger.info(
                    "Docstore strategy set to upserts, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            elif self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
                logger.info(
                    "Docstore strategy set to upserts and delete, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
            nodes_to_run = await self._ahandle_duplicates(
                input_nodes, store_doc_text=store_doc_text
            )

        else:
            nodes_to_run = input_nodes

        nodes = await arun_transformations(  # type: ignore
            nodes_to_run,
            self.transformations,
            show_progress=show_progress,
            **kwargs,
        )

        if self.vector_store is not None:
            nodes_with_embeddings = [n for n in nodes if n.embedding is not None]
            if nodes_with_embeddings:
                await self.vector_store.async_add(nodes_with_embeddings)

        if self.docstore is not None:
            await self._aupdate_docstore(nodes_to_run, store_doc_text=store_doc_text)

        return nodes
