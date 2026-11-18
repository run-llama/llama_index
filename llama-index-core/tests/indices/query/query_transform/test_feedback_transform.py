import pytest

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.core.llms import MockLLM


@pytest.mark.parametrize("used", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"evaluation": None}, {"evaluation": "invalid"}, {"evaluation": {}}],
)
def test_requires_evaluation_for_each_run(used, metadata):
    transform = FeedbackQueryTransformation(llm=MockLLM())
    if used:
        transform.run(
            "First question",
            {
                "evaluation": EvaluationResult(
                    response="First response", feedback="First feedback"
                )
            },
        )

    with pytest.raises(ValueError, match="Evaluation is not set"):
        transform.run("Second question", metadata)


@pytest.mark.parametrize("feedback", ["YES", "NO", "Check the source again"])
def test_uses_current_response_and_feedback(feedback):
    transform = FeedbackQueryTransformation(llm=MockLLM())
    transform.run(
        "First question",
        {
            "evaluation": EvaluationResult(
                response="First response", feedback="First feedback"
            )
        },
    )
    result = transform.run(
        "Second question",
        {"evaluation": EvaluationResult(response="Second response", feedback=feedback)},
    )

    assert "Second response" in result.query_str
    assert "Second question" in result.query_str
    assert "First response" not in result.query_str
    assert "First feedback" not in result.query_str
    assert result.custom_embedding_strs == ["Second question"]
    if feedback not in ("YES", "NO"):
        assert feedback in result.query_str


@pytest.mark.parametrize(
    "evaluation",
    [EvaluationResult(response="Answer"), EvaluationResult(feedback="Feedback")],
)
def test_incomplete_evaluation_is_rejected(evaluation):
    transform = FeedbackQueryTransformation(llm=MockLLM())

    with pytest.raises(ValueError, match="must contain response and feedback"):
        transform.run("Question", {"evaluation": evaluation})


def test_resynthesis_uses_current_evaluation(monkeypatch):
    transform = FeedbackQueryTransformation(llm=MockLLM(), resynthesize_query=True)
    calls = []

    def resynthesize(query, response, feedback):
        calls.append((query, response, feedback))
        return "Rewritten question"

    monkeypatch.setattr(transform, "_resynthesize_query", resynthesize)
    result = transform.run(
        "Question",
        {"evaluation": EvaluationResult(response="Answer", feedback="Needs revision")},
    )

    assert calls == [("Question", "Answer", "Needs revision")]
    assert result.query_str.endswith("Rewritten question")
    assert result.custom_embedding_strs == ["Question"]


def test_reentrant_resynthesis_keeps_evaluations_separate(monkeypatch):
    transform = FeedbackQueryTransformation(llm=MockLLM(), resynthesize_query=True)

    def resynthesize(query, response, feedback):
        if query == "Outer question":
            inner = transform.run(
                "Inner question",
                {
                    "evaluation": EvaluationResult(
                        response="Inner response", feedback="Inner feedback"
                    )
                },
            )
            assert "Inner response" in inner.query_str
        return query

    monkeypatch.setattr(transform, "_resynthesize_query", resynthesize)
    outer = transform.run(
        "Outer question",
        {
            "evaluation": EvaluationResult(
                response="Outer response", feedback="Outer feedback"
            )
        },
    )

    assert "Outer response" in outer.query_str
    assert "Outer feedback" in outer.query_str
    assert "Inner response" not in outer.query_str
    assert "Inner feedback" not in outer.query_str
