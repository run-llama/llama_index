"""Node parsers."""

from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.simple import SimpleNodeParser


__all__ = [
    "SimpleNodeParser",
    "NodeParser",
]
