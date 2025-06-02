import os

from collections import defaultdict
from enum import Enum
from tree_sitter import Node
from typing import Any, Dict, List, Optional, Sequence, Tuple


from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.extractors.metadata_extractors import BaseExtractor
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from llama_index.core.text_splitter import CodeSplitter
from llama_index.core.utils import get_tqdm_iterable


class _SignatureCaptureType(BaseModel):
    """
    Unfortunately some languages need special options for how to make a signature.

    For example, html element signatures should include their closing >, there is no
    easy way to include this using an always-exclusive system.

    However, using an always-inclusive system, python decorators don't work,
    as there isn't an easy to define terminator for decorators that is inclusive
    to their signature.
    """

    type: str = Field(description="The type string to match on.")
    inclusive: bool = Field(
        description=(
            "Whether to include the text of the node matched by this type or not."
        ),
    )


class _SignatureCaptureOptions(BaseModel):
    """
    Options for capturing the signature of a node.
    """

    start_signature_types: Optional[List[_SignatureCaptureType]] = Field(
        None,
        description=(
            "A list of node types any of which indicate the beginning of the signature."
            "If this is none or empty, use the start_byte of the node."
        ),
    )
    end_signature_types: Optional[List[_SignatureCaptureType]] = Field(
        None,
        description=(
            "A list of node types any of which indicate the end of the signature."
            "If this is none or empty, use the end_byte of the node."
        ),
    )
    name_identifier: str = Field(
        description=(
            "The node type to use for the signatures 'name'.If retrieving the name is"
            " more complicated than a simple type match, use a function which takes a"
            " node and returns true or false as to whether its the name or not. The"
            " first match is returned."
        )
    )


"""
Maps language -> Node Type -> SignatureCaptureOptions

The best way for a developer to discover these is to put a breakpoint at the TIP
tag in _chunk_node, and then create a unit test for some code, and then iterate
through the code discovering the node names.
"""
_DEFAULT_SIGNATURE_IDENTIFIERS: Dict[str, Dict[str, _SignatureCaptureOptions]] = {
    "python": {
        "function_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="block", inclusive=False)],
            name_identifier="identifier",
        ),
        "class_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="block", inclusive=False)],
            name_identifier="identifier",
        ),
    },
    "html": {
        "element": _SignatureCaptureOptions(
            start_signature_types=[_SignatureCaptureType(type="<", inclusive=True)],
            end_signature_types=[_SignatureCaptureType(type=">", inclusive=True)],
            name_identifier="tag_name",
        )
    },
    "cpp": {
        "class_specifier": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="type_identifier",
        ),
        "function_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="function_declarator",
        ),
    },
    "typescript": {
        "interface_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="type_identifier",
        ),
        "lexical_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="identifier",
        ),
        "function_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="identifier",
        ),
        "class_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="type_identifier",
        ),
        "method_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="property_identifier",
        ),
    },
    "php": {
        "function_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="}", inclusive=False)],
            name_identifier="name",
        ),
        "class_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="}", inclusive=False)],
            name_identifier="name",
        ),
        "method_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="}", inclusive=False)],
            name_identifier="name",
        ),
    },
}


class _ScopeMethod(Enum):
    INDENTATION = "INDENTATION"
    BRACKETS = "BRACKETS"
    HTML_END_TAGS = "HTML_END_TAGS"


class _CommentOptions(BaseModel):
    comment_template: str
    scope_method: _ScopeMethod


_COMMENT_OPTIONS: Dict[str, _CommentOptions] = {
    "cpp": _CommentOptions(
        comment_template="// {}", scope_method=_ScopeMethod.BRACKETS
    ),
    "html": _CommentOptions(
        comment_template="<!-- {} -->", scope_method=_ScopeMethod.HTML_END_TAGS
    ),
    "python": _CommentOptions(
        comment_template="# {}", scope_method=_ScopeMethod.INDENTATION
    ),
    "typescript": _CommentOptions(
        comment_template="// {}", scope_method=_ScopeMethod.BRACKETS
    ),
    "php": _CommentOptions(
        comment_template="// {}", scope_method=_ScopeMethod.BRACKETS
    ),
}

assert all(
    language in _DEFAULT_SIGNATURE_IDENTIFIERS for language in _COMMENT_OPTIONS
), "Not all languages in _COMMENT_OPTIONS are in _DEFAULT_SIGNATURE_IDENTIFIERS"
assert all(
    language in _COMMENT_OPTIONS for language in _DEFAULT_SIGNATURE_IDENTIFIERS
), "Not all languages in _DEFAULT_SIGNATURE_IDENTIFIERS are in _COMMENT_OPTIONS"


class _ScopeItem(BaseModel):
    """Like a Node from tree_sitter, but with only the str information we need."""

    name: str
    type: str
    signature: str


class _ChunkNodeOutput(BaseModel):
    """The output of a chunk_node call."""

    this_document: Optional[TextNode]
    upstream_children_documents: List[TextNode]
    all_documents: List[TextNode]


class CodeHierarchyNodeParser(NodeParser):
    """
    Split code using a AST parser.

    Add metadata about the scope of the code block and relationships between
    code blocks.
    """

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "CodeHierarchyNodeParser"

    language: str = Field(
        description="The programming language of the code being split."
    )
    signature_identifiers: Dict[str, _SignatureCaptureOptions] = Field(
        description=(
            "A dictionary mapping the type of a split mapped to the first and last type"
            " of itschildren which identify its signature."
        )
    )
    min_characters: int = Field(
        default=80,
        description=(
            "Minimum number of characters per chunk.Defaults to 80 because that's about"
            " how long a replacement comment is in skeleton mode."
        ),
    )
    code_splitter: Optional[CodeSplitter] = Field(
        description="The text splitter to use when splitting documents."
    )
    metadata_extractor: Optional[BaseExtractor] = Field(
        default=None, description="Metadata extraction pipeline to apply to nodes."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    skeleton: bool = Field(
        True,
        description=(
            "Parent nodes have the text of their child nodes replaced with a signature"
            " and a comment instructing the language model to visit the child node for"
            " the full text of the scope."
        ),
    )

    def __init__(
        self,
        language: str,
        skeleton: bool = True,
        signature_identifiers: Optional[Dict[str, _SignatureCaptureOptions]] = None,
        code_splitter: Optional[CodeSplitter] = None,
        callback_manager: Optional[CallbackManager] = None,
        metadata_extractor: Optional[BaseExtractor] = None,
        chunk_min_characters: int = 80,
    ):
        callback_manager = callback_manager or CallbackManager([])

        if signature_identifiers is None and language in _DEFAULT_SIGNATURE_IDENTIFIERS:
            signature_identifiers = _DEFAULT_SIGNATURE_IDENTIFIERS[language]

        super().__init__(
            include_prev_next_rel=False,
            language=language,
            callback_manager=callback_manager,
            metadata_extractor=metadata_extractor,
            code_splitter=code_splitter,
            signature_identifiers=signature_identifiers,
            min_characters=chunk_min_characters,
            skeleton=skeleton,
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
        return bytes(text, "utf-8")[start_byte:end_byte].decode().strip()

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
            _context_list (Optional[List[_ScopeItem]]): The scope context of the
                                                        parent node
            _root (bool): Whether or not this is the root node

        """
        if _context_list is None:
            _context_list = []

        upstream_children_documents: List[TextNode] = []
        all_documents: List[TextNode] = []

        # Capture any whitespace before parent.start_byte
        # Very important for space sensitive languages like python
        start_byte = parent.start_byte
        text_bytes = bytes(text, "utf-8")
        while start_byte > 0 and text_bytes[start_byte - 1 : start_byte] in (
            b" ",
            b"\t",
        ):
            start_byte -= 1

        # Create this node
        current_chunk = text_bytes[start_byte : parent.end_byte].decode()

        # Return early if the chunk is too small
        if len(current_chunk) < self.min_characters and not _root:
            return _ChunkNodeOutput(
                this_document=None, all_documents=[], upstream_children_documents=[]
            )

        # TIP: This is a wonderful place to put a debug breakpoint when
        #      Trying to integrate a new language. Pay attention to parent.type to learn
        #      all the available node types and their hierarchy.
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
                    "start_byte": start_byte,
                    "end_byte": parent.end_byte,
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

                # If there is a this_document, then we need
                # to add the children to this_document
                # and flush upstream_children_documents
                if this_document is not None:
                    # If we have been given a document, that means it's children
                    # are already set, so it needs to become a child of this node
                    if next_chunks.this_document is not None:
                        assert not next_chunks.upstream_children_documents, (
                            "next_chunks.this_document and"
                            " next_chunks.upstream_children_documents are exclusive."
                        )
                        this_document.relationships[NodeRelationship.CHILD].append(  # type: ignore
                            next_chunks.this_document.as_related_node_info()
                        )
                        next_chunks.this_document.relationships[
                            NodeRelationship.PARENT
                        ] = this_document.as_related_node_info()
                    # Otherwise, we have been given a list of
                    # upstream_children_documents. We need to make
                    # them a child of this node
                    else:
                        for d in next_chunks.upstream_children_documents:
                            this_document.relationships[NodeRelationship.CHILD].append(  # type: ignore
                                d.as_related_node_info()
                            )
                            d.relationships[NodeRelationship.PARENT] = (
                                this_document.as_related_node_info()
                            )
                # Otherwise we pass the children upstream
                else:
                    # If we have been given a document, that means it's
                    # children are already set, so it needs to become a
                    # child of the next node
                    if next_chunks.this_document is not None:
                        assert not next_chunks.upstream_children_documents, (
                            "next_chunks.this_document and"
                            " next_chunks.upstream_children_documents are exclusive."
                        )
                        upstream_children_documents.append(next_chunks.this_document)
                    # Otherwise, we have leftover children, they need
                    # to become children of the next node
                    else:
                        upstream_children_documents.extend(
                            next_chunks.upstream_children_documents
                        )

                # Lastly we need to maintain all documents
                all_documents.extend(next_chunks.all_documents)

        return _ChunkNodeOutput(
            this_document=this_document,
            upstream_children_documents=upstream_children_documents,
            all_documents=all_documents,
        )

    @staticmethod
    def get_code_hierarchy_from_nodes(
        nodes: Sequence[BaseNode],
        max_depth: int = -1,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Creates a code hierarchy appropriate to put into a tool description or context
        to make it easier to search for code.

        Call after `get_nodes_from_documents` and pass that output to this function.
        """
        out: Dict[str, Any] = defaultdict(dict)

        def get_subdict(keys: List[str]) -> Dict[str, Any]:
            # Get the dictionary we are operating on
            this_dict = out
            for key in keys:
                if key not in this_dict:
                    this_dict[key] = defaultdict(dict)
                this_dict = this_dict[key]
            return this_dict

        def recur_inclusive_scope(node: BaseNode, i: int, keys: List[str]) -> None:
            if "inclusive_scopes" not in node.metadata:
                raise KeyError("inclusive_scopes not in node.metadata")
            if i >= len(node.metadata["inclusive_scopes"]):
                return
            scope = node.metadata["inclusive_scopes"][i]

            this_dict = get_subdict(keys)

            if scope["name"] not in this_dict:
                this_dict[scope["name"]] = defaultdict(dict)

            if i < max_depth or max_depth == -1:
                recur_inclusive_scope(node, i + 1, [*keys, scope["name"]])

        def dict_to_markdown(d: Dict[str, Any], depth: int = 0) -> str:
            markdown = ""
            indent = "  " * depth  # Two spaces per depth level

            for key, value in d.items():
                if isinstance(value, dict):  # Check if value is a dict
                    # Add the key with a bullet point and increase depth for nested dicts
                    markdown += f"{indent}- {key}\n{dict_to_markdown(value, depth + 1)}"
                else:
                    # Handle non-dict items if necessary
                    markdown += f"{indent}- {key}: {value}\n"

            return markdown

        for node in nodes:
            filepath = node.metadata["filepath"].split("/")
            filepath[-1] = filepath[-1].split(".")[0]
            recur_inclusive_scope(node, 0, filepath)

        return out, dict_to_markdown(out)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        The main public method of this class.

        Parse documents into nodes.
        """
        out: List[BaseNode] = []

        try:
            import tree_sitter_language_pack
        except ImportError:
            raise ImportError(
                "Please install tree_sitter_language_pack to use CodeSplitter."
            )

        try:
            parser = tree_sitter_language_pack.get_parser(self.language)
            language = tree_sitter_language_pack.get_language(self.language)

            # Construct the path to the SCM file
            scm_fname = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pytree-sitter-queries",
                f"tree-sitter-{self.language}-tags.scm",
            )
        except Exception as e:
            print(
                f"Could not get parser for language {self.language}. Check "
                "https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages "
                "for a list of valid languages."
            )
            raise e  # noqa: TRY201

        query = None
        if self.signature_identifiers is None:
            assert os.path.exists(scm_fname), f"Could not find {scm_fname}"
            fp = open(scm_fname)
            query_scm = fp.read()
            query = language.query(query_scm)

        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing documents into nodes"
        )

        for node in nodes_with_progress:
            text = node.text
            tree = parser.parse(bytes(text, "utf-8"))

            if self.signature_identifiers is None:
                assert query is not None
                self.signature_identifiers = {}
                tag_to_type = {}
                captures = query.captures(tree.root_node)
                for _node, _tag in captures:
                    tag_to_type[_tag] = _node.type
                    if _tag.startswith("name.definition"):
                        # ignore name.
                        parent_tag = _tag[5:]
                        assert parent_tag in tag_to_type
                        parent_type = tag_to_type[parent_tag]
                        if parent_type not in self.signature_identifiers:
                            self.signature_identifiers[parent_type] = (
                                _SignatureCaptureOptions(name_identifier=_node.type)
                            )

            if (
                not tree.root_node.children
                or tree.root_node.children[0].type != "ERROR"
            ):
                # Chunk the code
                _chunks = self._chunk_node(tree.root_node, node.text)
                assert _chunks.this_document is not None, "Root node must be a chunk"
                chunks = _chunks.all_documents

                # Add your metadata to the chunks here
                for chunk in chunks:
                    chunk.metadata = {
                        "language": self.language,
                        **chunk.metadata,
                        **node.metadata,
                    }
                    chunk.relationships[NodeRelationship.SOURCE] = (
                        node.as_related_node_info()
                    )

                if self.skeleton:
                    self._skeletonize_list(chunks)

                # Now further split the code by lines and characters
                # TODO: Test this and the relationships it creates
                if self.code_splitter:
                    new_nodes = []
                    for original_node in chunks:
                        new_split_nodes = self.code_splitter.get_nodes_from_documents(
                            [original_node], show_progress=show_progress, **kwargs
                        )

                        if not new_split_nodes:
                            continue

                        # Force the first new_split_node to have the
                        # same id as the original_node
                        new_split_nodes[0].id_ = original_node.id_

                        # Add the UUID of the next node to the end of all nodes
                        for i, new_split_node in enumerate(new_split_nodes[:-1]):
                            new_split_node.text = (
                                new_split_node.text
                                + "\n"
                                + self._create_comment_line(new_split_nodes[i + 1], 0)
                            ).strip()

                        # Add the UUID of the previous node to the beginning of all nodes
                        for i, new_split_node in enumerate(new_split_nodes[1:]):
                            new_split_node.text = (
                                self._create_comment_line(new_split_nodes[i])
                                + new_split_node.text
                            ).strip()

                        # Add the parent child info to all the new_nodes_
                        # derived from node
                        for new_split_node in new_split_nodes:
                            new_split_node.relationships[NodeRelationship.CHILD] = (
                                original_node.child_nodes
                            )  # type: ignore
                            new_split_node.relationships[NodeRelationship.PARENT] = (
                                original_node.parent_node
                            )  # type: ignore

                        # Go through chunks and replace all
                        # instances of node.node_id in relationships
                        # with new_nodes_[0].node_id
                        for old_node in chunks:
                            # Handle child nodes, which are a list
                            new_children = []
                            for old_nodes_child in old_node.child_nodes or []:
                                if old_nodes_child.node_id == original_node.node_id:
                                    new_children.append(
                                        new_split_nodes[0].as_related_node_info()
                                    )
                                new_children.append(old_nodes_child)
                            old_node.relationships[NodeRelationship.CHILD] = (
                                new_children
                            )

                            # Handle parent node
                            if (
                                old_node.parent_node
                                and old_node.parent_node.node_id
                                == original_node.node_id
                            ):
                                old_node.relationships[NodeRelationship.PARENT] = (
                                    new_split_nodes[0].as_related_node_info()
                                )

                        # Now save new_nodes_
                        new_nodes += new_split_nodes

                    chunks = new_nodes

                # Or just extract metadata
                if self.metadata_extractor:
                    chunks = self.metadata_extractor.process_nodes(  # type: ignore
                        chunks
                    )

                out += chunks
            else:
                raise ValueError(f"Could not parse code with language {self.language}.")

        return out

    @staticmethod
    def _get_indentation(text: str) -> Tuple[str, int, int]:
        indent_char = None
        minimum_chain = None

        # Check that text is at least 1 line long
        text_split = text.splitlines()
        if len(text_split) == 0:
            raise ValueError("Text should be at least one line long.")

        for line in text_split:
            stripped_line = line.lstrip()

            if stripped_line:
                # Get whether it's tabs or spaces
                spaces_count = line.count(" ", 0, len(line) - len(stripped_line))
                tabs_count = line.count("\t", 0, len(line) - len(stripped_line))

                if not indent_char:
                    if spaces_count:
                        indent_char = " "
                    if tabs_count:
                        indent_char = "\t"

                # Detect mixed indentation.
                if spaces_count > 0 and tabs_count > 0:
                    raise ValueError("Mixed indentation found.")
                if indent_char == " " and tabs_count > 0:
                    raise ValueError("Mixed indentation found.")
                if indent_char == "\t" and spaces_count > 0:
                    raise ValueError("Mixed indentation found.")

                # Get the minimum chain of indent_char
                if indent_char:
                    char_count = line.count(
                        indent_char, 0, len(line) - len(stripped_line)
                    )
                    if minimum_chain is not None:
                        if char_count > 0:
                            minimum_chain = min(char_count, minimum_chain)
                    else:
                        if char_count > 0:
                            minimum_chain = char_count

        # Handle edge case
        if indent_char is None:
            indent_char = " "
        if minimum_chain is None:
            minimum_chain = 4

        # Get the first indent count
        first_line = text_split[0]
        first_indent_count = 0
        for char in first_line:
            if char == indent_char:
                first_indent_count += 1
            else:
                break

        # Return the default indent level if only one indentation level was found.
        return indent_char, minimum_chain, first_indent_count // minimum_chain

    @staticmethod
    def _get_comment_text(node: TextNode) -> str:
        """Gets just the natural language text for a skeletonize comment."""
        return f"Code replaced for brevity. See node_id {node.node_id}"

    @classmethod
    def _create_comment_line(cls, node: TextNode, indention_lvl: int = -1) -> str:
        """
        Creates a comment line for a node.

        Sometimes we don't use this in a loop because it requires recalculating
        a lot of the same information. But it is handy.
        """
        # Create the text to replace the child_node.text with
        language = node.metadata["language"]
        if language not in _COMMENT_OPTIONS:
            # TODO: Create a contribution message
            raise KeyError("Language not yet supported. Please contribute!")
        comment_options = _COMMENT_OPTIONS[language]
        (
            indentation_char,
            indentation_count_per_lvl,
            first_indentation_lvl,
        ) = cls._get_indentation(node.text)
        if indention_lvl != -1:
            first_indentation_lvl = indention_lvl
        else:
            first_indentation_lvl += 1
        return (
            indentation_char * indentation_count_per_lvl * first_indentation_lvl
            + comment_options.comment_template.format(cls._get_comment_text(node))
            + "\n"
        )

    @classmethod
    def _get_replacement_text(cls, child_node: TextNode) -> str:
        """
        Manufactures a the replacement text to use to skeletonize a given child node.
        """
        signature = child_node.metadata["inclusive_scopes"][-1]["signature"]
        language = child_node.metadata["language"]
        if language not in _COMMENT_OPTIONS:
            # TODO: Create a contribution message
            raise KeyError("Language not yet supported. Please contribute!")
        comment_options = _COMMENT_OPTIONS[language]

        # Create the text to replace the child_node.text with
        (
            indentation_char,
            indentation_count_per_lvl,
            first_indentation_lvl,
        ) = cls._get_indentation(child_node.text)

        # Start with a properly indented signature
        replacement_txt = (
            indentation_char * indentation_count_per_lvl * first_indentation_lvl
            + signature
        )

        # Add brackets if necessary. Expandable in the
        # future to other methods of scoping.
        if comment_options.scope_method == _ScopeMethod.BRACKETS:
            replacement_txt += " {\n"
            replacement_txt += (
                indentation_char
                * indentation_count_per_lvl
                * (first_indentation_lvl + 1)
                + comment_options.comment_template.format(
                    cls._get_comment_text(child_node)
                )
                + "\n"
            )
            replacement_txt += (
                indentation_char * indentation_count_per_lvl * first_indentation_lvl
                + "}"
            )

        elif comment_options.scope_method == _ScopeMethod.INDENTATION:
            replacement_txt += "\n"
            replacement_txt += indentation_char * indentation_count_per_lvl * (
                first_indentation_lvl + 1
            ) + comment_options.comment_template.format(
                cls._get_comment_text(child_node)
            )

        elif comment_options.scope_method == _ScopeMethod.HTML_END_TAGS:
            tag_name = child_node.metadata["inclusive_scopes"][-1]["name"]
            end_tag = f"</{tag_name}>"
            replacement_txt += "\n"
            replacement_txt += (
                indentation_char
                * indentation_count_per_lvl
                * (first_indentation_lvl + 1)
                + comment_options.comment_template.format(
                    cls._get_comment_text(child_node)
                )
                + "\n"
            )
            replacement_txt += (
                indentation_char * indentation_count_per_lvl * first_indentation_lvl
                + end_tag
            )

        else:
            raise KeyError(f"Unrecognized enum value {comment_options.scope_method}")

        return replacement_txt

    @classmethod
    def _skeletonize(cls, parent_node: TextNode, child_node: TextNode) -> None:
        """WARNING: In Place Operation."""
        # Simple protection clauses
        if child_node.text not in parent_node.text:
            raise ValueError("The child text is not contained inside the parent text.")
        if child_node.node_id not in (c.node_id for c in parent_node.child_nodes or []):
            raise ValueError("The child node is not a child of the parent node.")

        # Now do the replacement
        replacement_text = cls._get_replacement_text(child_node=child_node)

        index = parent_node.text.find(child_node.text)
        # If the text is found, replace only the first occurrence
        if index != -1:
            parent_node.text = (
                parent_node.text[:index]
                + replacement_text
                + parent_node.text[index + len(child_node.text) :]
            )

    @classmethod
    def _skeletonize_list(cls, nodes: List[TextNode]) -> None:
        # Create a convenient map for mapping node id's to nodes
        node_id_map = {n.node_id: n for n in nodes}

        def recur(node: TextNode) -> None:
            # If any children exist, skeletonize ourselves, starting at the root DFS
            for child in node.child_nodes or []:
                child_node = node_id_map[child.node_id]
                cls._skeletonize(parent_node=node, child_node=child_node)
                recur(child_node)

        # Iterate over root nodes and recur
        for n in nodes:
            if n.parent_node is None:
                recur(n)
