from typing import Dict, Type

from llama_index.legacy.node_parser.file.html import HTMLNodeParser
from llama_index.legacy.node_parser.file.json import JSONNodeParser
from llama_index.legacy.node_parser.file.markdown import MarkdownNodeParser
from llama_index.legacy.node_parser.file.simple_file import SimpleFileNodeParser
from llama_index.legacy.node_parser.interface import NodeParser
from llama_index.legacy.node_parser.relational.hierarchical import (
    HierarchicalNodeParser,
)
from llama_index.legacy.node_parser.text.code import CodeSplitter
from llama_index.legacy.node_parser.text.sentence import SentenceSplitter
from llama_index.legacy.node_parser.text.sentence_window import SentenceWindowNodeParser
from llama_index.legacy.node_parser.text.token import TokenTextSplitter

all_node_parsers: Dict[str, Type[NodeParser]] = {
    HTMLNodeParser.class_name(): HTMLNodeParser,
    JSONNodeParser.class_name(): JSONNodeParser,
    MarkdownNodeParser.class_name(): MarkdownNodeParser,
    SimpleFileNodeParser.class_name(): SimpleFileNodeParser,
    HierarchicalNodeParser.class_name(): HierarchicalNodeParser,
    CodeSplitter.class_name(): CodeSplitter,
    SentenceSplitter.class_name(): SentenceSplitter,
    TokenTextSplitter.class_name(): TokenTextSplitter,
    SentenceWindowNodeParser.class_name(): SentenceWindowNodeParser,
}


def load_parser(
    data: dict,
) -> NodeParser:
    if isinstance(data, NodeParser):
        return data
    parser_name = data.get("class_name", None)
    if parser_name is None:
        raise ValueError("Parser loading requires a class_name")

    if parser_name not in all_node_parsers:
        raise ValueError(f"Invalid parser name: {parser_name}")
    else:
        return all_node_parsers[parser_name].from_dict(data)
