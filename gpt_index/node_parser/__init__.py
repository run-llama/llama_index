"""Node parsers."""

from gpt_index.node_parser.interface import NodeParser
from gpt_index.node_parser.simple import SimpleNodeParser

__all__ = ["SimpleNodeParser", "NodeParser"]
