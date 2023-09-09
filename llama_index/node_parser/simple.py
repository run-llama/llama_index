"""Simple node parser."""
from typing import List, Optional, Sequence, Dict, Any

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.node_utils import get_nodes_from_document
from llama_index.schema import BaseNode, Document
from llama_index.text_splitter import (
    get_default_text_splitter,
    SplitterType,
    TextSplitter,
)
from llama_index.utils import get_tqdm_iterable


class _BaseSimpleNodeParser(NodeParser):
    """
    Base class for Simple node parser.

    Separated out the base class so that we can override the schema method
    on SimpleNodeParser to return a schema that has the text_splitter field
    as having the schema of TextSpliter instead of Union[TextSpliter, LC_TextSplitter].
    """

    text_splitter: TextSplitter = Field(
        description="The text splitter to use when splitting documents."
    )
    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    metadata_extractor: Optional[MetadataExtractor] = Field(
        default=None, description="Metadata extraction pipeline to apply to nodes."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )


class SimpleNodeParser(_BaseSimpleNodeParser):
    """Simple node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:
        text_splitter (Optional[TextSplitter]): text splitter
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    text_splitter: SplitterType = Field(  # type: ignore
        description="The text splitter to use when splitting documents."
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        text_splitter: Optional[SplitterType] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> "SimpleNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        text_splitter = text_splitter or get_default_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            callback_manager=callback_manager,
        )
        return cls(
            text_splitter=text_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "SimpleNodeParser"

    @classmethod
    def schema(cls, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Because text_splitter is of type Union[TextSpliter, LC_TextSplitter],
        we need to override the schema method to return a schema that has the
        text_splitter field as having the schema of TextSpliter instead
        of Union[TextSpliter, LC_TextSplitter].
        """
        super_schema = _BaseSimpleNodeParser.schema(**kwargs)
        super_schema["title"] = cls.class_name()
        super_schema["description"] = str(cls.__doc__).strip()
        return super_schema

    @classmethod
    def __modify_schema__(cls, schema: Dict[str, Any]) -> None:
        schema.update(properties=cls.schema()["properties"])

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_metadata (bool): whether to include metadata in nodes

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            for document in documents_with_progress:
                nodes = get_nodes_from_document(
                    document,
                    self.text_splitter,
                    self.include_metadata,
                    include_prev_next_rel=self.include_prev_next_rel,
                )
                all_nodes.extend(nodes)

            if self.metadata_extractor is not None:
                all_nodes = self.metadata_extractor.process_nodes(all_nodes)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes
