import os
import re
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional, Sequence, cast

from fsspec import AbstractFileSystem
from llama_index_client import (
    ConfigurableDataSourceNames,
    ConfigurableTransformationNames,
    ConfiguredTransformationItem,
    DataSourceCreate,
    Pipeline,
    PipelineCreate,
    Project,
    ProjectCreate,
)
from llama_index_client.client import PlatformApi

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from llama_index.ingestion.data_sources import (
    ConfigurableDataSources,
    ConfiguredDataSource,
)
from llama_index.ingestion.transformations import (
    ConfigurableTransformations,
    ConfiguredTransformation,
)
from llama_index.node_parser import SentenceSplitter
from llama_index.readers.base import ReaderConfig
from llama_index.schema import (
    BaseComponent,
    BaseNode,
    Document,
    MetadataMode,
    TransformComponent,
)
from llama_index.service_context import ServiceContext
from llama_index.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.storage.storage_context import DOCSTORE_FNAME
from llama_index.utils import concat_dirs
from llama_index.vector_stores.types import BasePydanticVectorStore

DEFAULT_PIPELINE_NAME = "default"
DEFAULT_PROJECT_NAME = "default"
DEFAULT_BASE_URL = "https://api.prod.llamaindex.ai"
DEFAULT_APP_URL = "https://prod.llamaindex.ai"


def deserialize_transformation_component(
    component_dict: dict, component_type: ConfigurableTransformationNames
) -> BaseComponent:
    component_cls = ConfigurableTransformations[component_type].value.component_type
    return component_cls.from_dict(component_dict)


def deserialize_source_component(
    component_dict: dict, component_type: ConfigurableDataSourceNames
) -> BaseComponent:
    component_cls = ConfigurableDataSources[component_type].value.component_type
    return component_cls.from_dict(component_dict)


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


class DocstoreStrategy(str, Enum):
    """Document de-duplication strategy."""

    UPSERTS = "upserts"
    DUPLICATES_ONLY = "duplicates_only"


class IngestionPipeline(BaseModel):
    """An ingestion pipeline that can be applied to data."""

    name: str = Field(
        default=DEFAULT_PIPELINE_NAME,
        description="Unique name of the ingestion pipeline",
    )
    project_name: str = Field(
        default=DEFAULT_PROJECT_NAME, description="Unique name of the project"
    )

    transformations: List[TransformComponent] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    readers: List[ReaderConfig] = Field(
        description="Readers to use to read the data", default_factor=list
    )
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

    platform_base_url: str = Field(
        default=DEFAULT_BASE_URL, description="Base URL for the platform API"
    )
    platform_app_url: str = Field(
        default=DEFAULT_APP_URL, description="Base URL for the platform app"
    )
    platform_api_key: Optional[str] = Field(
        default=None, description="Platform API key"
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        transformations: Optional[List[TransformComponent]] = None,
        readers: Optional[List[ReaderConfig]] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        cache: Optional[IngestionCache] = None,
        disable_cache: bool = False,
        platform_base_url: Optional[str] = None,
        platform_app_url: Optional[str] = None,
        platform_api_key: Optional[str] = None,
        docstore: Optional[BaseDocumentStore] = None,
        docstore_strategy: DocstoreStrategy = DocstoreStrategy.UPSERTS,
    ) -> None:
        if transformations is None:
            transformations = self._get_default_transformations()

        platform_base_url = platform_base_url or os.environ.get(
            "PLATFORM_BASE_URL", DEFAULT_BASE_URL
        )
        platform_app_url = platform_app_url or os.environ.get(
            "PLATFORM_APP_URL", DEFAULT_APP_URL
        )
        platform_api_key = platform_api_key or os.environ.get("PLATFORM_API_KEY", None)

        super().__init__(
            name=name,
            project_name=project_name,
            transformations=transformations,
            readers=readers or [],
            documents=documents,
            vector_store=vector_store,
            cache=cache or IngestionCache(),
            disable_cache=disable_cache,
            platform_base_url=platform_base_url,
            platform_api_key=platform_api_key,
            platform_app_url=platform_app_url,
            docstore=docstore,
            docstore_strategy=docstore_strategy,
        )

    @classmethod
    def from_service_context(
        cls,
        service_context: ServiceContext,
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        readers: Optional[List[ReaderConfig]] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        cache: Optional[IngestionCache] = None,
        docstore: Optional[BaseDocumentStore] = None,
    ) -> "IngestionPipeline":
        transformations = [
            *service_context.transformations,
            service_context.embed_model,
        ]

        return cls(
            name=name,
            project_name=project_name,
            transformations=transformations,
            readers=readers or [],
            documents=documents,
            vector_store=vector_store,
            cache=cache,
            docstore=docstore,
        )

    @classmethod
    def from_pipeline_name(
        cls,
        name: str,
        project_name: str = DEFAULT_PROJECT_NAME,
        platform_base_url: Optional[str] = None,
        cache: Optional[IngestionCache] = None,
        platform_api_key: Optional[str] = None,
        platform_app_url: Optional[str] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        disable_cache: bool = False,
    ) -> "IngestionPipeline":
        platform_base_url = platform_base_url or os.environ.get(
            "PLATFORM_BASE_URL", DEFAULT_BASE_URL
        )
        assert platform_base_url is not None

        platform_api_key = platform_api_key or os.environ.get("PLATFORM_API_KEY", None)
        platform_app_url = platform_app_url or os.environ.get("PLATFORM_APP_URL", None)

        client = PlatformApi(base_url=platform_base_url, token=platform_api_key)

        projects: List[Project] = client.project.list_projects(
            project_name=project_name
        )
        if len(projects) < 0:
            raise ValueError(f"Project with name {project_name} not found")

        project = projects[0]
        assert project.id is not None, "Project ID should not be None"

        pipelines: List[Pipeline] = client.pipeline.get_pipeline_by_name(
            pipeline_name=name, project_name=project_name
        )
        if len(pipelines) < 0:
            raise ValueError(f"Pipeline with name {name} not found")

        pipeline = pipelines[0]

        transformations: List[TransformComponent] = []
        for configured_transformation in pipeline.configured_transformations:
            component_dict = cast(dict, configured_transformation.component)
            transformation_component_type = (
                configured_transformation.configurable_transformation_type
            )
            transformation = deserialize_transformation_component(
                component_dict, transformation_component_type
            )
            transformations.append(transformation)

        documents = []
        readers = []
        for data_source in pipeline.data_sources:
            component_dict = cast(dict, data_source.component)
            source_component_type = data_source.source_type
            source_component = deserialize_source_component(
                component_dict, source_component_type
            )

            if data_source.source_type == ConfigurableDataSourceNames.READER:
                readers.append(source_component)
            elif (
                data_source.source_type == ConfigurableDataSourceNames.DOCUMENT
                and isinstance(source_component, BaseNode)
                and source_component.get_content()
            ):
                documents.append(source_component)

        return cls(
            name=name,
            project_name=project_name,
            transformations=transformations,
            readers=readers,
            documents=documents,
            vector_store=vector_store,
            platform_base_url=platform_base_url,
            cache=cache,
            disable_cache=disable_cache,
            platform_api_key=platform_api_key,
            platform_app_url=platform_app_url,
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

    def register(
        self,
        verbose: bool = True,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
    ) -> str:
        client = PlatformApi(
            base_url=self.platform_base_url, token=self.platform_api_key
        )

        input_nodes = self._prepare_inputs(documents, nodes)

        configured_transformations: List[ConfiguredTransformation] = []
        for transformation in self.transformations:
            try:
                configured_transformations.append(
                    ConfiguredTransformation.from_component(transformation)
                )
            except ValueError:
                raise ValueError(f"Unsupported transformation: {type(transformation)}")

        configured_transformation_items: List[ConfiguredTransformationItem] = []
        for item in configured_transformations:
            name = ConfigurableTransformationNames[
                item.configurable_transformation_type.name
            ]
            configured_transformation_items.append(
                ConfiguredTransformationItem(
                    transformation_name=name,
                    component=item.component,
                    configurable_transformation_type=item.configurable_transformation_type.name,
                )
            )

            # remove callback manager
            configured_transformation_items[-1].component.pop("callback_manager", None)  # type: ignore

        data_sources = []
        for reader in self.readers:
            if reader.reader.is_remote:
                configured_data_source = ConfiguredDataSource.from_component(
                    reader,
                )
                source_type = ConfigurableDataSourceNames[
                    configured_data_source.configurable_data_source_type.name
                ]
                data_sources.append(
                    DataSourceCreate(
                        name=configured_data_source.name,
                        source_type=source_type,
                        component=configured_data_source.component,
                    )
                )
            else:
                documents = reader.read()
                input_nodes += documents

        for node in input_nodes:
            configured_data_source = ConfiguredDataSource.from_component(node)
            source_type = ConfigurableDataSourceNames[
                configured_data_source.configurable_data_source_type.name
            ]
            data_sources.append(
                DataSourceCreate(
                    name=configured_data_source.name,
                    source_type=source_type,
                    component=node,
                )
            )

        project = client.project.upsert_project(
            request=ProjectCreate(name=self.project_name)
        )
        assert project.id is not None, "Project ID should not be None"

        # upload
        pipeline = client.project.upsert_pipeline_for_project(
            project.id,
            request=PipelineCreate(
                name=self.name,
                configured_transformations=configured_transformation_items,
                data_sources=data_sources,
                data_sinks=[],
            ),
        )
        assert pipeline.id is not None, "Pipeline ID should not be None"

        # Print playground URL if not running remote
        if verbose:
            print(
                f"Pipeline available at: {self.platform_app_url}/project/{project.id}/playground/{pipeline.id}"
            )

        return pipeline.id

    def run_remote(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
    ) -> str:
        client = PlatformApi(
            base_url=self.platform_base_url, token=self.platform_api_key
        )

        pipeline_id = self.register(documents=documents, nodes=nodes, verbose=False)

        # start pipeline?
        # the `PipeLineExecution` object should likely generate a URL at some point
        pipeline_execution = client.pipeline.create_configured_transformation_execution(
            pipeline_id
        )

        assert (
            pipeline_execution.id is not None
        ), "Pipeline execution ID should not be None"

        print(
            f"Find your remote results here: {self.platform_app_url}/"
            f"pipelines/execution?id={pipeline_execution.id}"
        )

        return pipeline_execution.id

    def _prepare_inputs(
        self,
        documents: Optional[List[Document]],
        nodes: Optional[List[BaseNode]],
        read_local: bool = False,
    ) -> List[Document]:
        input_nodes: List[BaseNode] = []
        if documents is not None:
            input_nodes += documents

        if nodes is not None:
            input_nodes += nodes

        if self.documents is not None:
            input_nodes += self.documents

        for reader in self.readers:
            if read_local:
                input_nodes += reader.read()

        return input_nodes

    def _handle_duplicates(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Handle docstore duplicates by checking all hashes."""
        assert self.docstore is not None

        existing_hashes = self.docstore.get_all_document_hashes()
        current_hashes = []
        nodes_to_run = []
        for node in nodes:
            if node.hash not in existing_hashes and node.hash not in current_hashes:
                self.docstore.add_documents([node])
                self.docstore.set_document_hash(node.id_, node.hash)
                nodes_to_run.append(node)
                current_hashes.append(node.hash)
        return nodes_to_run

    def _handle_upserts(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None

        deduped_nodes_to_run = {}
        for node in nodes:
            ref_doc_id = node.ref_doc_id if node.ref_doc_id else node.id_

            existing_hash = self.docstore.get_document_hash(ref_doc_id)
            if not existing_hash:
                # document doesn't exist, so add it
                self.docstore.add_documents([node])
                self.docstore.set_document_hash(ref_doc_id, node.hash)
                deduped_nodes_to_run[ref_doc_id] = node
            elif existing_hash and existing_hash != node.hash:
                self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

                self.docstore.add_documents([node])
                self.docstore.set_document_hash(ref_doc_id, node.hash)

                deduped_nodes_to_run[ref_doc_id] = node
            else:
                continue  # document exists and is unchanged, so skip it

        return list(deduped_nodes_to_run.values())

    def run(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        input_nodes = self._prepare_inputs(documents, nodes, read_local=True)

        # check if we need to dedup
        if self.docstore is not None and self.vector_store is not None:
            if self.docstore_strategy == DocstoreStrategy.UPSERTS:
                nodes_to_run = self._handle_upserts(input_nodes)
            elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
                nodes_to_run = self._handle_duplicates(input_nodes)
            else:
                raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
        elif self.docstore is not None and self.vector_store is None:
            if self.docstore_strategy == DocstoreStrategy.UPSERTS:
                print(
                    "Docstore strategy set to upserts, but no vector store. "
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
        input_nodes = self._prepare_inputs(documents, nodes, read_local=True)

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
