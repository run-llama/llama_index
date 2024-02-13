from enum import Enum
from pathlib import Path
from typing import Any, Generic, Iterable, List, Optional, Type, TypeVar, cast

from unique_names_generator import get_random_name
from unique_names_generator.data import (
    ADJECTIVES,
    ANIMALS,
    COLORS,
)

from llama_index.bridge.pydantic import BaseModel, Field, GenericModel
from llama_index.readers import (
    BeautifulSoupWebReader,
    DiscordReader,
    ElasticsearchReader,
    NotionPageReader,
    RssReader,
    SimpleWebPageReader,
    SlackReader,
    TrafilaturaWebReader,
    TwitterTweetReader,
    WikipediaReader,
    YoutubeTranscriptReader,
)
from llama_index.readers.base import BasePydanticReader, ReaderConfig
from llama_index.readers.google_readers.gdocs import GoogleDocsReader
from llama_index.readers.google_readers.gsheets import GoogleSheetsReader
from llama_index.schema import BaseComponent, Document, TextNode

# used for generating random names for data sources
name_combo = [ADJECTIVES, COLORS, ANIMALS]


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


class ConfigurableDataSources(Enum):
    """
    Enumeration of all supported DataSource instances.
    """

    DOCUMENT = DataSource(
        name="Document",
        component_type=Document,
    )

    TEXT_NODE = DataSource(
        name="TextNode",
        component_type=TextNode,
    )

    DISCORD = DataSource(
        name="Discord",
        component_type=DiscordReader,
    )

    ELASTICSEARCH = DataSource(
        name="Elasticsearch",
        component_type=ElasticsearchReader,
    )

    NOTION_PAGE = DataSource(
        name="Notion Page",
        component_type=NotionPageReader,
    )

    SLACK = DataSource(
        name="Slack",
        component_type=SlackReader,
    )

    TWITTER = DataSource(
        name="Twitter",
        component_type=TwitterTweetReader,
    )

    SIMPLE_WEB_PAGE = DataSource(
        name="Simple Web Page",
        component_type=SimpleWebPageReader,
    )

    TRAFILATURA_WEB_PAGE = DataSource(
        name="Trafilatura Web Page",
        component_type=TrafilaturaWebReader,
    )

    BEAUTIFUL_SOUP_WEB_PAGE = DataSource(
        name="Beautiful Soup Web Page",
        component_type=BeautifulSoupWebReader,
    )

    RSS = DataSource(
        name="RSS",
        component_type=RssReader,
    )

    WIKIPEDIA = DataSource(
        name="Wikipedia",
        component_type=WikipediaReader,
    )

    YOUTUBE_TRANSCRIPT = DataSource(
        name="Youtube Transcript",
        component_type=YoutubeTranscriptReader,
    )

    GOOGLE_DOCS = DataSource(
        name="Google Docs",
        component_type=GoogleDocsReader,
    )

    GOOGLE_SHEETS = DataSource(
        name="Google Sheets",
        component_type=GoogleSheetsReader,
    )

    READER = DataSource(
        name="Reader",
        component_type=ReaderConfig,
    )

    DOCUMENT_GROUP = DataSource(
        name="Document Group",
        component_type=DocumentGroup,
    )

    @classmethod
    def from_component(cls, component: BaseComponent) -> "ConfigurableDataSources":
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
            reader_config = ReaderConfig(loader=component)
            return ConfiguredDataSource[ReaderConfig](
                component=reader_config
            )  # type: ignore

        if isinstance(component, DocumentGroup) and name is None:
            # if the component is a DocumentGroup, we want to use the
            # full file path as the name of the data source
            component = cast(DocumentGroup, component)
            name = component.file_path

        if name is None:
            suffix = get_random_name(combo=name_combo, separator="-", style="lowercase")
            name = self.value.name + f" [{suffix}]]"
        return ConfiguredDataSource[component_type](  # type: ignore
            component=component, name=name
        )


T = TypeVar("T", bound=BaseComponent)


class ConfiguredDataSource(GenericModel, Generic[T]):
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
    def configurable_data_source_type(self) -> ConfigurableDataSources:
        return ConfigurableDataSources.from_component(self.component)
