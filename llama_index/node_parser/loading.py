from typing import Optional

from llama_index.node_parser.extractors.metadata_extractors import MetadataExtractor
from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.sentence_window import SentenceWindowNodeParser
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.text_splitter.types import SplitterType


def load_parser(
    data: dict,
    text_splitter: Optional[SplitterType] = None,
    metadata_extractor: Optional[MetadataExtractor] = None,
) -> NodeParser:
    parser_name = data.get("class_name", None)
    if parser_name is None:
        raise ValueError("Parser loading requires a class_name")

    if parser_name == SimpleNodeParser.class_name():
        return SimpleNodeParser.from_dict(
            data, text_splitter=text_splitter, metadata_extractor=metadata_extractor
        )
    elif parser_name == SentenceWindowNodeParser.class_name():
        assert isinstance(text_splitter, (type(None), SentenceSplitter))
        return SentenceWindowNodeParser.from_dict(
            data, sentence_splitter=text_splitter, metadata_extractor=metadata_extractor
        )
    else:
        raise ValueError(f"Unknown parser name: {parser_name}")
