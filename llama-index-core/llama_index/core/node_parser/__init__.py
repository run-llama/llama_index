"""Node parsers."""

from llama_index.core.node_parser.file.html import HTMLNodeParser
from llama_index.core.node_parser.file.json import JSONNodeParser
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from llama_index.core.node_parser.file.simple_file import SimpleFileNodeParser
from llama_index.core.node_parser.interface import (
    MetadataAwareTextSplitter,
    NodeParser,
    TextSplitter,
)
from llama_index.core.node_parser.relational.hierarchical import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
    get_child_nodes,
    get_deeper_nodes,
)
from llama_index.core.node_parser.relational.markdown_element import (
    MarkdownElementNodeParser,
)
from llama_index.core.node_parser.relational.unstructured_element import (
    UnstructuredElementNodeParser,
)
from llama_index.core.node_parser.text.code import CodeSplitter
from llama_index.core.node_parser.text.langchain import LangchainNodeParser
from llama_index.core.node_parser.text.semantic_splitter import (
    SemanticSplitterNodeParser,
)
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.node_parser.text.sentence_window import (
    SentenceWindowNodeParser,
)
from llama_index.core.node_parser.text.token import TokenTextSplitter

# deprecated, for backwards compatibility
SimpleNodeParser = SentenceSplitter

__all__ = [
    "TokenTextSplitter",
    "SentenceSplitter",
    "CodeSplitter",
    "SimpleFileNodeParser",
    "HTMLNodeParser",
    "MarkdownNodeParser",
    "JSONNodeParser",
    "SentenceWindowNodeParser",
    "SemanticSplitterNodeParser",
    "NodeParser",
    "HierarchicalNodeParser",
    "TextSplitter",
    "MarkdownElementNodeParser",
    "MetadataAwareTextSplitter",
    "LangchainNodeParser",
    "UnstructuredElementNodeParser",
    "get_leaf_nodes",
    "get_root_nodes",
    "get_child_nodes",
    "get_deeper_nodes",
    # deprecated, for backwards compatibility
    "SimpleNodeParser",
]
