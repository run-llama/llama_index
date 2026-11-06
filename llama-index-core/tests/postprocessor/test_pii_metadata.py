"""Test metadata rendering after PII postprocessing."""

import json
import sys
from types import ModuleType
from typing import Any, Type

import pytest

from llama_index.core.base.llms.types import TextBlock
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.pii import (
    NERPIINodePostprocessor,
    PIINodePostprocessor,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    BaseNode,
    MediaResource,
    MetadataMode,
    Node,
    NodeWithScore,
    TextNode,
)

ORIGINAL_EMAIL = "alice@example.com"
HIDDEN_EMAIL = "private@example.com"


@pytest.fixture(params=["llm", "ner"])
def postprocessor(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> BaseNodePostprocessor:
    if request.param == "llm":

        def predict(self: MockLLM, prompt: Any, **kwargs: Any) -> str:
            text = kwargs["context_str"]
            masked = text.replace(ORIGINAL_EMAIL, "[EMAIL]")
            token_to_original = (
                {"[EMAIL]": ORIGINAL_EMAIL} if ORIGINAL_EMAIL in text else {}
            )
            return f"{masked}\nOutput Mapping:\n{json.dumps(token_to_original)}"

        monkeypatch.setattr(MockLLM, "predict", predict)
        return PIINodePostprocessor(llm=MockLLM())

    def ner(text: str) -> list:
        if ORIGINAL_EMAIL not in text:
            return []
        start = text.index(ORIGINAL_EMAIL)
        return [
            {
                "entity_group": "EMAIL",
                "start": start,
                "end": start + len(ORIGINAL_EMAIL),
                "word": ORIGINAL_EMAIL,
            }
        ]

    transformers = ModuleType("transformers")
    monkeypatch.setattr(
        transformers, "pipeline", lambda *args, **kwargs: ner, raising=False
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return NERPIINodePostprocessor()


@pytest.mark.parametrize(
    ("node_type", "text"),
    [
        pytest.param(TextNode, "Contact support", id="text-node"),
        pytest.param(Node, "Contact support", id="node"),
        pytest.param(Node, None, id="node-without-text"),
    ],
)
def test_pii_metadata_is_not_rendered_again_for_llm(
    postprocessor: BaseNodePostprocessor,
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
    token_to_original = output.metadata["__pii_node_info__"]
    assert list(token_to_original.values()) == [ORIGINAL_EMAIL]
    email_token = next(iter(token_to_original))
    expected = f"source: contacts.txt\nemail: {email_token}"
    if text:
        expected += f"\n\n{text}"

    rendered = output.get_content(metadata_mode=MetadataMode.LLM)
    assert rendered == expected
    assert ORIGINAL_EMAIL not in rendered
    assert output.get_content_blocks(metadata_mode=MetadataMode.LLM) == [
        TextBlock(text=expected)
    ]

    # Raw metadata stays available for filtering and source attribution.
    assert output.metadata == {**metadata, "__pii_node_info__": token_to_original}
    assert original.model_dump() == snapshot
