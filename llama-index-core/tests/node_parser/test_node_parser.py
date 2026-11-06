from typing import Any, List, Sequence
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode, Document, NodeRelationship


class _TestNodeParser(NodeParser):
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return super()._parse_nodes(nodes, show_progress, **kwargs)


def test__postprocess_parsed_nodes_include_metadata():
    np = _TestNodeParser()

    nodes = []
    for i in range(3):
        node = TextNode(text=f"I am Node number {i}")
        node.metadata = {"node_number": i}
        nodes.append(node)

    ret = np._postprocess_parsed_nodes(nodes, {})
    for i, node in enumerate(ret):
        assert node.metadata == {"node_number": i}


def test__postprocess_parsed_nodes_include_metadata_parent_doc():
    np = _TestNodeParser()
    doc = Document(text="I am root")
    doc.metadata = {"document_type": "root"}

    nodes = []
    for i in range(3):
        node = TextNode(text=f"I am Node number {i}")
        node.metadata = {"node_number": i}
        node.relationships = {NodeRelationship.SOURCE: doc.as_related_node_info()}
        nodes.append(node)

    ret = np._postprocess_parsed_nodes(nodes, {})
    for i, node in enumerate(ret):
        assert node.metadata == {"node_number": i, "document_type": "root"}


def test_node_parser_serialization_round_trips_default_id_func() -> None:
    # #23986: a node parser with the default id_func must load back from its
    # serialized form instead of failing pydantic's callable validation.
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.node_parser.loading import load_parser
    from llama_index.core.node_parser.node_utils import default_id_func

    config = SentenceSplitter(chunk_size=128, chunk_overlap=0).to_dict()
    restored = load_parser(config)

    assert isinstance(restored, SentenceSplitter)
    assert restored.id_func is default_id_func
