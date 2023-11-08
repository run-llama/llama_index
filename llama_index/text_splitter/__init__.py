# TODO: Deprecated import support for old text splitters
from llama_index.node_parser.text.code import CodeSplitter as CodeSplitter
from llama_index.node_parser.text.sentence import (
    SentenceSplitter as SentenceSplitter,
)
from llama_index.node_parser.text.token import TokenTextSplitter as TokenTextSplitter

__all__ = [
    "SentenceSplitter",
    "TokenTextSplitter",
    "CodeSplitter",
]
