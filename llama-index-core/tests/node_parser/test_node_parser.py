import pytest
from typing import Any, List, Sequence
from llama_index.core.node_parser import (
    MetadataAwareTextSplitter,
    NodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import BaseNode, TextNode, Document, NodeRelationship


class _TestNodeParser(NodeParser):
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return super()._parse_nodes(nodes, show_progress, **kwargs)


class _TestMetadataAwareSplitter(MetadataAwareTextSplitter):
    """Carries the metadata-aware machinery without a real splitting strategy."""

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        return [text]

    def split_text(self, text: str) -> List[str]:
        return [text]


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


def test__get_metadata_str_is_empty_when_metadata_is_disabled():
    """
    Disabled metadata must not reserve split budget.

    With `include_metadata=False` the nodes will not carry metadata, so
    reserving budget for it can only reject text that fits.
    """
    np = _TestMetadataAwareSplitter(include_metadata=False)
    doc = Document(text="word " * 12, metadata={"context": "x" * 100})

    assert np._get_metadata_str(doc) == ""


def test__get_metadata_str_still_reports_metadata_when_enabled():
    """
    The opt-out must not silence the oversized-metadata guard.

    The guard stays correct when metadata is wanted, so an oversized field is
    still rejected once the opt-out stops reserving budget for it.
    """
    np = _TestMetadataAwareSplitter(include_metadata=True)
    doc = Document(text="word " * 12, metadata={"context": "x" * 100})

    assert len(np._get_metadata_str(doc)) > 100


@pytest.mark.parametrize(
    "splitter_cls", [TokenTextSplitter, SentenceSplitter], ids=["token", "sentence"]
)
def test_disabled_metadata_does_not_block_ingestion(splitter_cls):
    """
    A large metadata field must not fail ingestion when opted out of.

    Both splitters shared `MetadataAwareTextSplitter._get_metadata_str()`, which
    formatted the metadata regardless of the opt-out and then had the budget
    check reject the text that was otherwise within budget.
    """
    splitter = splitter_cls(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=False
    )

    nodes = splitter.get_nodes_from_documents(
        [Document(text="word " * 12, metadata={"context": "x" * 100})]
    )

    assert len(nodes) > 0
    assert all(node.metadata == {} for node in nodes)


@pytest.mark.parametrize(
    "splitter_cls", [TokenTextSplitter, SentenceSplitter], ids=["token", "sentence"]
)
def test_enabled_metadata_still_rejects_an_oversized_field(splitter_cls):
    """The control: an oversized field is still an error when metadata is wanted."""
    splitter = splitter_cls(
        chunk_size=20, chunk_overlap=0, tokenizer=list, include_metadata=True
    )

    with pytest.raises(ValueError, match="Metadata length"):
        splitter.get_nodes_from_documents(
            [Document(text="word " * 12, metadata={"context": "x" * 100})]
        )
