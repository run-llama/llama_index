"""Test query transform."""

from llama_index.core.indices.query.query_transform.base import (
    DecomposeQueryTransform,
    StepBackQueryTransform,
)
from tests.indices.query.query_transform.mock_utils import (
    MOCK_DECOMPOSE_PROMPT,
    MOCK_STEPBACK_PROMPT,
)


def test_decompose_query_transform(patch_llm_predictor) -> None:
    """Test decompose query transform."""
    query_transform = DecomposeQueryTransform(
        decompose_query_prompt=MOCK_DECOMPOSE_PROMPT
    )

    query_str = "What is?"
    new_query_bundle = query_transform.run(query_str, {"index_summary": "Foo bar"})
    assert new_query_bundle.query_str == "What is?:Foo bar"
    assert new_query_bundle.embedding_strs == ["What is?:Foo bar"]


def test_step_back_query_transform(patch_llm_predictor) -> None:
    """Test step-back query transform."""
    query_transform = StepBackQueryTransform(step_back_prompt=MOCK_STEPBACK_PROMPT)

    query_str = "What school did Alice attend between Aug and Nov 1954?"
    new_query_bundle = query_transform.run(query_str)
    assert (
        new_query_bundle.query_str
        == "step-back(What school did Alice attend between Aug and Nov 1954?)"
    )
    assert new_query_bundle.embedding_strs == [
        "step-back(What school did Alice attend between Aug and Nov 1954?)"
    ]
