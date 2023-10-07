# TODO: Deprecated import support for old text splitters
from llama_index.node_parser.text.code import CodeNodeParser as CodeSplitter
from llama_index.node_parser.text.sentence import (
    SentenceAwareNodeParser as SentenceSplitter,
)
from llama_index.node_parser.text.token import TokenAwareNodeParser as TokenTextSplitter

__all__ = [
    "SentenceSplitter",
    "TokenTextSplitter",
    "CodeSplitter",
]
