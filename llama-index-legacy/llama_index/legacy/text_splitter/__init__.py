# TODO: Deprecated import support for old text splitters
from llama_index.legacy.node_parser.text.code import CodeSplitter
from llama_index.legacy.node_parser.text.sentence import (
    SentenceSplitter,
)
from llama_index.legacy.node_parser.text.token import TokenTextSplitter

__all__ = [
    "SentenceSplitter",
    "TokenTextSplitter",
    "CodeSplitter",
]
