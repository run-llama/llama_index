from llama_index.node_parser.interface import NodeParser
from llama_index.node_parser.relational.sentence_window import SentenceWindowNodeParser
from llama_index.node_parser.relational.simple import SimpleNodeParser


def load_parser(
    data: dict,
) -> NodeParser:
    if isinstance(data, NodeParser):
        return data
    parser_name = data.get("class_name", None)
    if parser_name is None:
        raise ValueError("Parser loading requires a class_name")

    if parser_name == SimpleNodeParser.class_name():
        return SimpleNodeParser.from_dict(
            data,
        )
    elif parser_name == SentenceWindowNodeParser.class_name():
        return SentenceWindowNodeParser.from_dict(
            data,
        )
    else:
        raise ValueError(f"Unknown parser name: {parser_name}")
