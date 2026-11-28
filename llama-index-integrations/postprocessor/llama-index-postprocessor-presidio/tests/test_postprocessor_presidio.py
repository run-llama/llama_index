from typing import Any, Type
from unittest.mock import Mock

import pytest
from presidio_analyzer import RecognizerResult

from llama_index.core.base.llms.types import TextBlock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    BaseNode,
    MediaResource,
    MetadataMode,
    Node,
    NodeWithScore,
    TextNode,
)
from llama_index.postprocessor.presidio import PresidioPIINodePostprocessor

ORIGINAL_EMAIL = "alice@example.com"
HIDDEN_EMAIL = "private@example.com"


def test_class():
    names_of_base_classes = [b.__name__ for b in PresidioPIINodePostprocessor.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


@pytest.fixture
def postprocessor(monkeypatch: pytest.MonkeyPatch) -> PresidioPIINodePostprocessor:
    detector = Mock()

    def analyze(*, text: str, **kwargs: Any) -> list:
        if ORIGINAL_EMAIL not in text:
            return []
        start = text.index(ORIGINAL_EMAIL)
        return [
            RecognizerResult("EMAIL_ADDRESS", start, start + len(ORIGINAL_EMAIL), 1.0)
        ]

    detector.analyze.side_effect = analyze
    monkeypatch.setattr(
        "llama_index.postprocessor.presidio.base.AnalyzerEngine", lambda: detector
    )
    return PresidioPIINodePostprocessor(presidio_entities=["EMAIL_ADDRESS"])


@pytest.mark.parametrize(
    ("node_type", "text"),
    [
        pytest.param(TextNode, "Contact support", id="text-node"),
        pytest.param(Node, "Contact support", id="node"),
        pytest.param(Node, None, id="node-without-text"),
    ],
)
def test_presidio_metadata_is_not_rendered_again_for_llm(
    postprocessor: PresidioPIINodePostprocessor,
    node_type: Type[BaseNode],
    text: str | None,
) -> None:
    metadata = {
        "source": "contacts.txt",
        "email": ORIGINAL_EMAIL,
        "hidden_email": HIDDEN_EMAIL,
    }
    kwargs: dict[str, Any] = {
        "metadata": metadata,
        "excluded_llm_metadata_keys": ["hidden_email"],
    }
    if node_type is TextNode:
        original = TextNode(text=text or "", **kwargs)
    else:
        text_resource = MediaResource(text=text) if text is not None else None
        original = Node(text_resource=text_resource, **kwargs)
    snapshot = original.model_dump()
    output = postprocessor.postprocess_nodes([NodeWithScore(node=original)])[0].node
    expected = "source: contacts.txt\nemail: <EMAIL_ADDRESS_1>"
    if text:
        expected += f"\n\n{text}"

    assert output.get_content(metadata_mode=MetadataMode.LLM) == expected
    assert output.get_content_blocks(metadata_mode=MetadataMode.LLM) == [
        TextBlock(text=expected)
    ]
    assert output.metadata == {
        **metadata,
        "__pii_node_info__": {"<EMAIL_ADDRESS_1>": ORIGINAL_EMAIL},
    }
    assert original.model_dump() == snapshot
