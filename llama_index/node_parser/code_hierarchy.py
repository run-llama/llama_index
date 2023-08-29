
from typing import Any, List, Optional, Sequence

from pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.schema import BaseNode, Document, NodeRelationship, TextNode
from llama_index.text_splitter.code_splitter import CodeSplitter
from llama_index.utils import get_tqdm_iterable


class CodeBlockNodeParser(NodeParser):
    """Split code using a AST parser.

    Add metadata about the scope of the code block and relationships between
    code blocks.
    """

    language: str = Field(
        description="The programming languge of the code being split."
    )
    split_on_types: List[str] = Field(
        description="The types of nodes to split on."
    )
    code_splitter: Optional[CodeSplitter] = Field(
        description="The text splitter to use when splitting documents."
    )
    metadata_extractor: Optional[MetadataExtractor] = Field(
        default=None, description="Metadata extraction pipeline to apply to nodes."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    def __init__(
        self,
        language: str,
        split_on_types: List[str],
        code_splitter: Optional[CodeSplitter] = None,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ):
        callback_manager = callback_manager or CallbackManager([])
        super().__init__(
            language=language,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
            split_on_types=split_on_types,
            code_splitter=code_splitter,
        )

    def _chunk_node(self, parent: Any, text: str, context_list: Optional[List[str]] = None) -> List[TextNode]:
        if context_list is None:
            context_list = []

        new_nodes: List[TextNode] = []

        # We will assemble the context string from the context list
        # For anything other than the last context, we will add an ellipses comment
        # to indicate that there is more context before the current chunk
        context_str = "".join(context_list)

        current_chunk = str(context_str)  # Initialize current_chunk with current context

        last_child = None
        for child in parent.children:
            breakpoint()
            # Add the new signature or header to the context list before recursing
            if child.type in self.split_on_types:
                if len(current_chunk) > 0:  # If current_chunk has more than just the context
                    # Stop the current chunk
                    new_node = TextNode(
                        text=current_chunk,
                        relationships={
                            NodeRelationship.PARENT: parent,
                            NodeRelationship.CHILD: [],
                            NodeRelationship.NEXT: None,
                            NodeRelationship.PREVIOUS: last_child.as_related_node_info() if last_child is not None else None,
                        }
                    )
                    new_nodes.append(new_node)
                else:
                    new_node = None

                # Create a new chunk recursively
                new_context = text[last_child.start_bytes:last_child.end_bytes]
                context_list.append(new_context)
                next_chunks = self._chunk_node(child, text, context_list=context_list.copy())

                # Create relationship heirarchy
                if new_node is not None:
                    new_node.relationships[NodeRelationship.CHILD] = [chunk.as_related_node_info() for chunk in next_chunks]
                    for chunk in next_chunks:
                        chunk.relationships[NodeRelationship.PARENT] = new_node.as_related_node_info()

                new_nodes.extend(next_chunks)
            else:
                current_chunk += text[child.start_bytes:child.end_byte]


            last_child = child

        if len(current_chunk) > 0:  # If current_chunk has more than just the context
            new_node = TextNode(
                text=current_chunk,
                relationships={
                    NodeRelationship.PARENT: None,
                    NodeRelationship.CHILD: [],
                    NodeRelationship.NEXT: None,
                    NodeRelationship.PREVIOUS: None,
                }
            )
            new_nodes.append(new_node)

        # Add ordered relationships between chunks
        for i in range(len(new_nodes)):
            if i > 0:
                new_nodes[i].relationships[NodeRelationship.PREVIOUS] = new_nodes[i-1].as_related_node_info()
            if i < len(new_nodes) - 1:
                new_nodes[i].relationships[NodeRelationship.NEXT] = new_nodes[i+1].as_related_node_info()

        return new_nodes

        @abstractmethod
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse

        """

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            try:
                import tree_sitter_languages
            except ImportError:
                raise ImportError(
                    "Please install tree_sitter_languages to use CodeSplitter."
                )

            try:
                parser = tree_sitter_languages.get_parser(self.language)
            except Exception as e:
                print(
                    f"Could not get parser for language {self.language}. Check "
                    "https://github.com/grantjenks/py-tree-sitter-languages#license "
                    "for a list of valid languages."
                )
                raise e

            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )
            for document in documents_with_progress:
                text = document.text
                tree = parser.parse(bytes(text, "utf-8"))

                if (
                    not tree.root_node.children
                    or tree.root_node.children[0].type != "ERROR"
                ):
                    # Chunk the code
                    chunks = self._chunk_node(tree.root_node, document.text)

                    # Add your metadata to the chunks here
                    for chunk in chunks:
                        chunk.metadata = {
                            "language": self.language,
                            **document.metadata,
                        }
                        chunk.relationships[NodeRelationship.SOURCE] = document.as_related_node_info()

                    # Now further split the code by lines and characters
                    if self.code_splitter:
                        simple = SimpleNodeParser(
                            text_splitter=self.code_splitter,
                            include_metadata=True,
                            include_prev_next_rel=True,
                            metadata_extractor=self.metadata_extractor,
                            callback_manager=self.callback_manager,
                        )
                        chunks = simple.process_nodes(chunks)

                    # Or just extract metadata
                    elif self.metadata_extractor:
                        chunks = self.metadata_extractor.process_nodes(chunks)

                    event.on_end(
                        payload={EventPayload.CHUNKS: chunks},
                    )

                    return chunks
                else:
                    raise ValueError(f"Could not parse code with language {self.language}.")


