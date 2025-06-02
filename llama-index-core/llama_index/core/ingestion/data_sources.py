import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Iterable, List, Optional, Type, TypeVar, cast

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
    ValidationError,
)
from llama_index.core.readers.base import BasePydanticReader, ReaderConfig
from llama_index.core.schema import BaseComponent, Document, TextNode


class DataSource(BaseModel):
    """
    A class containing metadata for a type of data source.
    """

    name: str = Field(
        description="Unique and human-readable name for the type of data source"
    )
    component_type: Type[BaseComponent] = Field(
        description="Type of component that implements the data source"
    )


class DocumentGroup(BasePydanticReader):
    """
    A group of documents, usually separate pages from a single file.
    """

    file_path: str = Field(description="Path to the file containing the documents")
    documents: List[Document] = Field(
        description="Sequential group of documents, usually separate pages from a single file."
    )

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name

    @classmethod
    def class_name(cls) -> str:
        return "DocumentGroup"

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        return self.documents


class ConfigurableComponent(Enum):
    @classmethod
    def from_component(cls, component: BaseComponent) -> "ConfigurableComponent":
        component_class = type(component)
        for component_type in cls:
            if component_type.value.component_type == component_class:
                return component_type
        raise ValueError(
            f"Component {component} is not a supported data source component."
        )

    def build_configured_data_source(
        self, component: BaseComponent, name: Optional[str] = None
    ) -> "ConfiguredDataSource":
        component_type = self.value.component_type
        if not isinstance(component, component_type):
            raise ValueError(
                f"The enum value {self} is not compatible with component of "
                f"type {type(component)}"
            )
        elif isinstance(component, BasePydanticReader):
            reader_config = ReaderConfig(reader=component)
            return ConfiguredDataSource[ReaderConfig](component=reader_config)  # type: ignore

        if isinstance(component, DocumentGroup) and name is None:
            # if the component is a DocumentGroup, we want to use the
            # full file path as the name of the data source
            component = cast(DocumentGroup, component)
            name = component.file_path

        if name is None:
            suffix = uuid.uuid1()
            name = self.value.name + f" [{suffix}]]"
        return ConfiguredDataSource[component_type](  # type: ignore
            component=component, name=name
        )


def build_configurable_data_source_enum() -> ConfigurableComponent:
    """
    Build an enum of configurable data sources.
    But conditional on if the corresponding reader is available.
    """
    enum_members = []

    try:
        from llama_index.readers.discord import DiscordReader  # pants: no-infer-dep

        enum_members.append(
            (
                "DISCORD",
                DataSource(
                    name="Discord",
                    component_type=DiscordReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.elasticsearch import (
            ElasticsearchReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "ELASTICSEARCH",
                DataSource(
                    name="Elasticsearch",
                    component_type=ElasticsearchReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.notion import NotionPageReader  # pants: no-infer-dep

        enum_members.append(
            (
                "NOTION_PAGE",
                DataSource(
                    name="Notion Page",
                    component_type=NotionPageReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.slack import SlackReader  # pants: no-infer-dep

        enum_members.append(
            (
                "SLACK",
                DataSource(
                    name="Slack",
                    component_type=SlackReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.twitter import (
            TwitterTweetReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "TWITTER",
                DataSource(
                    name="Twitter",
                    component_type=TwitterTweetReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.web import SimpleWebPageReader  # pants: no-infer-dep

        enum_members.append(
            (
                "SIMPLE_WEB_PAGE",
                DataSource(
                    name="Simple Web Page",
                    component_type=SimpleWebPageReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.web import TrafilaturaWebReader  # pants: no-infer-dep

        enum_members.append(
            (
                "TRAFILATURA_WEB_PAGE",
                DataSource(
                    name="Trafilatura Web Page",
                    component_type=TrafilaturaWebReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.web import (
            BeautifulSoupWebReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "BEAUTIFUL_SOUP_WEB_PAGE",
                DataSource(
                    name="Beautiful Soup Web Page",
                    component_type=BeautifulSoupWebReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.web import RssReader  # pants: no-infer-dep

        enum_members.append(
            (
                "RSS",
                DataSource(
                    name="RSS",
                    component_type=RssReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.wikipedia import WikipediaReader  # pants: no-infer-dep

        enum_members.append(
            (
                "WIKIPEDIA",
                DataSource(
                    name="Wikipedia",
                    component_type=WikipediaReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.youtube_transcript import (
            YoutubeTranscriptReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "YOUTUBE_TRANSCRIPT",
                DataSource(
                    name="Youtube Transcript",
                    component_type=YoutubeTranscriptReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.google import GoogleDocsReader  # pants: no-infer-dep

        enum_members.append(
            (
                "GOOGLE_DOCS",
                DataSource(
                    name="Google Docs",
                    component_type=GoogleDocsReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.google import GoogleSheetsReader  # pants: no-infer-dep

        enum_members.append(
            (
                "GOOGLE_SHEETS",
                DataSource(
                    name="Google Sheets",
                    component_type=GoogleSheetsReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.s3 import S3Reader  # pants: no-infer-dep

        enum_members.append(
            (
                "S3",
                DataSource(
                    name="S3",
                    component_type=S3Reader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.azstorage_blob import (
            AzStorageBlobReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "AZURE_STORAGE_BLOB",
                DataSource(
                    name="Azure Storage Blob",
                    component_type=AzStorageBlobReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.gcs import GCSReader  # pants: no-infer-dep

        enum_members.append(
            (
                "GCS",
                DataSource(
                    name="GCS",
                    component_type=GCSReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.google import GoogleDriveReader  # pants: no-infer-dep

        enum_members.append(
            (
                "GOOGLE_DRIVE",
                DataSource(
                    name="Google Drive",
                    component_type=GoogleDriveReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.microsoft_onedrive import (
            OneDriveReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "MICROSOFT_ONEDRIVE",
                DataSource(
                    name="Microsoft OneDrive",
                    component_type=OneDriveReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    try:
        from llama_index.readers.microsoft_sharepoint import (
            SharePointReader,
        )  # pants: no-infer-dep

        enum_members.append(
            (
                "MICROSOFT_SHAREPOINT",
                DataSource(
                    name="Microsoft Sharepoint",
                    component_type=SharePointReader,
                ),
            )
        )
    except (ImportError, ValidationError):
        pass

    enum_members.append(
        (
            "READER",
            DataSource(
                name="Reader",
                component_type=ReaderConfig,
            ),
        )
    )

    enum_members.append(
        (
            "DOCUMENT_GROUP",
            DataSource(
                name="Document Group",
                component_type=DocumentGroup,
            ),
        )
    )

    enum_members.append(
        (
            "TEXT_NODE",
            DataSource(
                name="Text Node",
                component_type=TextNode,
            ),
        )
    )

    enum_members.append(
        (
            "DOCUMENT",
            DataSource(
                name="Document",
                component_type=Document,
            ),
        )
    )

    return ConfigurableComponent("ConfigurableDataSources", enum_members)  # type: ignore


ConfigurableDataSources = build_configurable_data_source_enum()

T = TypeVar("T", bound=BaseComponent)


class ConfiguredDataSource(BaseModel, Generic[T]):
    """
    A class containing metadata & implementation for a data source in a pipeline.
    """

    name: str
    component: T = Field(description="Component that implements the data source")

    @classmethod
    def from_component(
        cls, component: BaseComponent, name: Optional[str] = None
    ) -> "ConfiguredDataSource":
        """
        Build a ConfiguredDataSource from a component.

        This should be the preferred way to build a ConfiguredDataSource
        as it will ensure that the component is supported as indicated by having a
        corresponding enum value in DataSources.

        This has the added bonus that you don't need to specify the generic type
        like ConfiguredDataSource[Document]. The return value of
        this ConfiguredDataSource.from_component(document) will be
        ConfiguredDataSource[Document] if document is
        a Document object.
        """
        return ConfigurableDataSources.from_component(
            component
        ).build_configured_data_source(component, name)

    @property
    def configurable_data_source_type(self) -> ConfigurableComponent:
        return ConfigurableDataSources.from_component(self.component)
