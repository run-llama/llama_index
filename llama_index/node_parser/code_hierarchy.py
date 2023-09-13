from typing import Any, Dict, Iterable, Callable, List, Optional, Sequence, Tuple

from pydantic import Field
from tree_sitter import Node
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.schema import BaseNode, Document, NodeRelationship, TextNode
from llama_index.text_splitter.code_splitter import CodeSplitter
from llama_index.utils import get_tqdm_iterable
from pydantic import BaseModel


class SignatureType(BaseModel):
    """
    Unfortunately some languages need special options for how to make a signature.

    For example, html element signatures should include their closing >, there is no
    easy way to include this using an always-exclusive system.

    However, using an always-inclusive system, python decorators don't work, as there isn't
    an easy to define terminator for decorators that is inclusive to their signature.
    """

    type: str = Field(description="The type string to match on.")
    inclusive: bool = Field(
        description="Whether to include the text of the node matched by this type or not.",
    )


class SignatureOptions(BaseModel):
    start_signature_types: Optional[List[SignatureType]] = Field(
        None,
        description="A list of node types any of which indicate the beginning of the signature."
        "If this is none or empty, use the start_byte of the node.",
    )
    end_signature_types: Optional[List[SignatureType]] = Field(
        None,
        description="A list of node types any of which indicate the end of the signature."
        "If this is none or empty, use the end_byte of the node.",
    )
    name_identifier: str = Field(
        description="The node type to use for the signatures 'name'."
        "If retrieving the name is more complicated than a simple type match, use a function which "
        "takes a node and returns true or false as to whether its the name or not. "
        "The first match is returned."
    )


DEFAULT_SIGNATURE_IDENTIFIERS: Dict[str, Dict[str, SignatureOptions]] = {
    "python": {
        "function_definition": SignatureOptions(
            end_signature_types=[SignatureType(type="block", inclusive=False)],
            name_identifier="identifier",
        ),
        "class_definition": SignatureOptions(
            end_signature_types=[SignatureType(type="block", inclusive=False)],
            name_identifier="identifier",
        ),
    },
    "html": {
        "element": SignatureOptions(
            start_signature_types=[SignatureType(type="<", inclusive=True)],
            end_signature_types=[SignatureType(type=">", inclusive=True)],
            name_identifier="tag_name",
        )
    },
    "cpp": {
        "class_specifier": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="type_identifier",
        ),
        "function_definition": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="function_declarator",
        ),
    },
    "typescript": {
        "interface_declaration": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="type_identifier",
        ),
        "lexical_declaration": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="identifier",
        ),
        "function_declaration": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="identifier",
        ),
        "class_declaration": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="type_identifier",
        ),
        "method_definition": SignatureOptions(
            end_signature_types=[SignatureType(type="{", inclusive=False)],
            name_identifier="property_identifier",
        ),
    },
}


ELLIPSES_COMMENT = " ... May have additional code availible in other documents, cut for brevity ..."  # The comment to use when chunking code


def _generate_comment_line(language: str, comment: str) -> str:
    """
    Generates a comment line in a given language.
    Able to handle languages that require closing comment symbols as well.
    If the language isn't recognized, uses 🦙 emojis.
    """
    single_line = {
        "Ada": "--",
        "Agda": "--",
        "Apex": "//",
        "Bash": "#",
        "C": "//",
        "C++": "//",
        "C#": "//",
        "Clojure": ";;",
        "CMake": "#",
        "Common Lisp": ";;",
        "CUDA": "//",
        "Dart": "//",
        "D": "//",
        "Dockerfile": "#",
        "Elixir": "#",
        "Elm": "--",
        "Emacs Lisp": ";;",
        "Erlang": "%",
        "Fish": "#",
        "Formula": "#",  # Assuming Excel-like formula
        "Fortran": "!",
        "Go": "//",
        "Graphql": "#",
        "Hack": "//",
        "Haskell": "--",
        "HCL": "#",
        "HTML": "<!-- {} -->",
        "Java": "//",
        "JavaScript": "//",
        "jq": "#",
        "JSON5": "//",
        "Julia": "#",
        "Kotlin": "//",
        "Latex": "%",
        "Lua": "--",
        "Make": "#",
        "Markdown": "<!-- {} -->",  # Technically HTML but often used in Markdown
        "Meson": "#",
        "Nix": "#",
        "Objective-C": "//",
        "OCaml": "(* {} *)",
        "Pascal": "//",
        "Perl": "#",
        "PHP": "//",
        "PowerShell": "#",
        "Protocol Buffers": "//",
        "Python": "#",
        "QML": "//",
        "R": "#",
        "Ruby": "#",
        "Rust": "//",
        "Scala": "//",
        "Scheme": ";",
        "Scss": "//",
        "Shell": "#",
        "SQL": "--",
        "Svelte": "<!-- {} -->",  # HTML-like
        "Swift": "//",
        "TOML": "#",
        "Turtle": "#",
        "TypeScript": "//",
        "Verilog": "//",
        "VHDL": "--",
        "Vue": "//",
        "WASM": ";;",
        "YAML": "#",
        "YANG": "//",
        "Zig": "//",
    }
    single_line = {k.lower(): v for k, v in single_line.items()}

    if language.lower() in single_line:
        syntax = single_line[language.lower()]
        if "{}" in syntax:
            return syntax.format(comment)
        else:
            return f"{syntax} {comment}"
    else:
        return f"🦙{comment}🦙"  # For unknown languages, use emoji to enclose the comment


class _ScopeItem(BaseModel):
    """Like a Node from tree_sitter, but with only the str information we need."""

    name: str
    type: str
    signature: str


class CodeHierarchyNodeParser(NodeParser):
    """Split code using a AST parser.

    Add metadata about the scope of the code block and relationships between
    code blocks.
    """

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return cls.__name__

    language: str = Field(
        description="The programming languge of the code being split."
    )
    signature_identifiers: Dict[str, SignatureOptions] = Field(
        description=(
            "A dictionary mapping the type of a split mapped to the first and last type of its"
            "children which identify its signature."
        )
    )
    min_characters: int = Field(
        default=0, description="Minimum number of characters per chunk."
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
        signature_identifiers: Optional[Dict[str, SignatureOptions]] = None,
        code_splitter: Optional[CodeSplitter] = None,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        min_characters: int = 0,
    ):
        callback_manager = callback_manager or CallbackManager([])

        if signature_identifiers is None:
            try:
                signature_identifiers = DEFAULT_SIGNATURE_IDENTIFIERS[language]
            except KeyError:
                raise ValueError(
                    f"Must provide signature_identifiers for language {language}."
                )

        super().__init__(
            language=language,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
            code_splitter=code_splitter,
            signature_identifiers=signature_identifiers,
            min_characters=min_characters,
        )

    def _get_node_name(self, node: Node) -> str:
        """Get the name of a node."""
        signature_identifier = self.signature_identifiers[node.type]

        def recur(node: Node) -> str:
            for child in node.children:
                if child.type == signature_identifier.name_identifier:
                    return child.text.decode()
                if child.children:
                    out = recur(child)
                    if out:
                        return out
            return ""

        return recur(node).strip()

    def _get_node_signature(self, text: str, node: Node) -> str:
        """Get the signature of a node."""
        signature_identifier = self.signature_identifiers[node.type]

        def find_start(node: Node) -> Optional[int]:
            if not signature_identifier.start_signature_types:
                signature_identifier.start_signature_types = []

            for st in signature_identifier.start_signature_types:
                if node.type == st.type:
                    if st.inclusive:
                        return node.start_byte
                    return node.end_byte

            for child in node.children:
                out = find_start(child)
                if out is not None:
                    return out

            return None

        def find_end(node: Node) -> Optional[int]:
            if not signature_identifier.end_signature_types:
                signature_identifier.end_signature_types = []

            for st in signature_identifier.end_signature_types:
                if node.type == st.type:
                    if st.inclusive:
                        return node.end_byte
                    return node.start_byte

            for child in node.children:
                out = find_end(child)
                if out is not None:
                    return out

            return None

        start_byte, end_byte = find_start(node), find_end(node)
        if start_byte is None:
            start_byte = node.start_byte
        if end_byte is None:
            end_byte = node.end_byte
        return text[start_byte:end_byte].strip()

    class _ChunkNodeOutput(BaseModel):
        """The output of a chunk_node call."""

        this_document: Optional[TextNode]
        upstream_children_documents: List[TextNode]
        all_documents: List[TextNode]

    def _chunk_node(
        self,
        parent: Node,
        text: str,
        _context_list: Optional[List[_ScopeItem]] = None,
        _root: bool = True,
    ) -> _ChunkNodeOutput:
        """
        This is really the "main" method of this class. It is recursive and recursively
        chunks the text by the options identified in self.signature_identifiers.

        It is ran by get_nodes_from_documents.

        Args:
            parent (Node): The parent node to chunk
            text (str): The text of the entire document
            _context_list (Optional[List[_ScopeItem]]): The scope context of the parent node
            _root (bool): Whether or not this is the root node
        """
        if _context_list is None:
            _context_list = []

        upstream_children_documents: List[TextNode] = []
        all_documents: List[TextNode] = []

        # Capture any whitespace before parent.start_byte
        # Very important for space sensitive languages like python
        start_byte = parent.start_byte
        while start_byte > 0 and text[start_byte - 1] in (" ", "\t"):
            start_byte -= 1

        # Create this node
        current_chunk = text[start_byte : parent.end_byte]

        # Return early if the chunk is too small
        if len(current_chunk) < self.min_characters and not _root:
            return self._ChunkNodeOutput(
                this_document=None, all_documents=[], upstream_children_documents=[]
            )

        # TIP: This is a wonderful place to put a debug breakpoint when
        #      Trying to integrate a new language. Pay attention to parent.type to learn
        #      all the availible node types and their hierarchy.
        if parent.type in self.signature_identifiers or _root:
            # Get the new context
            if not _root:
                new_context = _ScopeItem(
                    name=self._get_node_name(parent),
                    type=parent.type,
                    signature=self._get_node_signature(text=text, node=parent),
                )
                _context_list.append(new_context)
            this_document = TextNode(
                text=current_chunk,
                metadata={
                    "inclusive_scopes": [cl.dict() for cl in _context_list],
                },
                relationships={
                    NodeRelationship.CHILD: [],
                },
            )
            all_documents.append(this_document)
        else:
            this_document = None

        # Iterate over children
        for child in parent.children:
            if child.children:
                # Recurse on the child
                next_chunks = self._chunk_node(
                    child, text, _context_list=_context_list.copy(), _root=False
                )

                # If there is a this_document, then we need to add the children to this_document
                # and flush upstream_children_documents
                if this_document is not None:
                    # If we have been given a document, that means it's children are already set, so it needs to become a child of this node
                    if next_chunks.this_document is not None:
                        assert (
                            not next_chunks.upstream_children_documents
                        ), "next_chunks.this_document and next_chunks.upstream_children_documents are exclusive."
                        this_document.relationships[NodeRelationship.CHILD].append(
                            next_chunks.this_document.as_related_node_info()
                        )
                        next_chunks.this_document.relationships[
                            NodeRelationship.PARENT
                        ] = this_document.as_related_node_info()
                    # Otherwise, we have been given a list of upstream_children_documents. We need to make them a child of this node
                    else:
                        for d in next_chunks.upstream_children_documents:
                            this_document.relationships[NodeRelationship.CHILD].append(
                                d.as_related_node_info()
                            )
                            d.relationships[
                                NodeRelationship.PARENT
                            ] = this_document.as_related_node_info()
                # Otherwise we pass the children upstream
                else:
                    # If we have been given a document, that means it's children are already set, so it needs to become a child of the next node
                    if next_chunks.this_document is not None:
                        assert (
                            not next_chunks.upstream_children_documents
                        ), "next_chunks.this_document and next_chunks.upstream_children_documents are exclusive."
                        upstream_children_documents.append(next_chunks.this_document)
                    # Otherwise, we have leftover children, they need to become children of the next node
                    else:
                        upstream_children_documents.extend(
                            next_chunks.upstream_children_documents
                        )

                # Lastly we need to maintain all documents
                all_documents.extend(next_chunks.all_documents)

        return self._ChunkNodeOutput(
            this_document=this_document,
            upstream_children_documents=upstream_children_documents,
            all_documents=all_documents,
        )

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """
        The main public method of this class.

        Parse documents into nodes.
        """
        out: List[BaseNode] = []
        with self.callback_manager.event(
            CBEventType.CHUNKING,
            payload={EventPayload.CHUNKS: [document.text for document in documents]},
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
                    _chunks = self._chunk_node(tree.root_node, document.text)
                    assert (
                        _chunks.this_document is not None
                    ), "Root node must be a chunk"
                    chunks = _chunks.all_documents

                    # Add your metadata to the chunks here
                    for chunk in chunks:
                        chunk.metadata = {
                            "language": self.language,
                            **chunk.metadata,
                            **document.metadata,
                        }
                        chunk.relationships[
                            NodeRelationship.SOURCE
                        ] = document.as_related_node_info()

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

                    out += chunks
                else:
                    raise ValueError(
                        f"Could not parse code with language {self.language}."
                    )

        return out
