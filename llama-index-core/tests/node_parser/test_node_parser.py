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


def test_id_func_serialization_roundtrip():
    """A parser using the default id_func must survive to_dict/from_dict."""
    import pytest as _pytest

    from llama_index.core.node_parser import SentenceSplitter

    parser = SentenceSplitter(chunk_size=1024)
    data = parser.to_dict()

    restored = SentenceSplitter.from_dict(data)
    assert restored.id_func is parser.id_func
    assert restored.to_dict()["id_func"] == data["id_func"]


def test_id_func_serialization_unknown_name_fails_clearly():
    """A serialized non-default id_func must fail with a clear error."""
    import pytest as _pytest

    from llama_index.core.node_parser import SentenceSplitter

    data = SentenceSplitter(chunk_size=1024).to_dict()
    data["id_func"] = {"id_func_name": "some_custom_func", "title": "id_func"}

    with _pytest.raises(ValueError, match="only the default id_func round-trips"):
        SentenceSplitter.from_dict(data)
