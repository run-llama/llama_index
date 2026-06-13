"""Tests for HotpotQA benchmark evaluator."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.evaluation.benchmarks.hotpotqa import (
    HotpotQAEvaluator,
    HotpotQARetriever,
    exact_match_score,
    f1_score,
    normalize_answer,
)
from llama_index.core.schema import QueryBundle


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_answer_lowercase():
    assert normalize_answer("Hello World") == "hello world"


def test_normalize_answer_removes_articles():
    assert normalize_answer("The cat sat on a mat") == "cat sat on mat"


def test_normalize_answer_removes_punctuation():
    assert normalize_answer("Hello, World!") == "hello world"


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer("  hello   world  ") == "hello world"


# ---------------------------------------------------------------------------
# exact_match_score
# ---------------------------------------------------------------------------


def test_exact_match_score_true():
    assert exact_match_score("The Cat", "the cat") is True


def test_exact_match_score_false():
    assert exact_match_score("dog", "cat") is False


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------


def test_f1_score_perfect_match():
    f1, precision, recall = f1_score("quick brown fox", "quick brown fox")
    assert f1 == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_f1_score_no_overlap():
    result = f1_score("cat", "dog")
    assert result == (0, 0, 0)


def test_f1_score_partial_overlap():
    f1, precision, recall = f1_score("quick brown fox", "quick brown dog")
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)


def test_f1_score_yes_no_mismatch():
    assert f1_score("yes", "no") == (0, 0, 0)


def test_f1_score_no_answer():
    assert f1_score("something else", "noanswer") == (0, 0, 0)


# ---------------------------------------------------------------------------
# HotpotQARetriever
# ---------------------------------------------------------------------------

SAMPLE_QUERY_OBJECTS = [
    {
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare",
        "context": [
            ["Source A", ["Shakespeare wrote Hamlet.", "It is a tragedy."]],
            ["Source B", ["Hamlet is a Danish prince."]],
        ],
    }
]


def test_hotpotqa_retriever_returns_nodes():
    retriever = HotpotQARetriever(SAMPLE_QUERY_OBJECTS)
    bundle = QueryBundle(
        query_str="Who wrote Hamlet?",
        custom_embedding_strs=["Who wrote Hamlet?"],
    )
    results = retriever._retrieve(bundle)
    assert len(results) == 2
    assert all(r.score == 1.0 for r in results)


def test_hotpotqa_retriever_uses_query_str_fallback():
    retriever = HotpotQARetriever(SAMPLE_QUERY_OBJECTS)
    bundle = QueryBundle(query_str="Who wrote Hamlet?")
    results = retriever._retrieve(bundle)
    assert len(results) == 2


def test_hotpotqa_retriever_str():
    retriever = HotpotQARetriever(SAMPLE_QUERY_OBJECTS)
    assert str(retriever) == "HotpotQARetriever"


def test_hotpotqa_retriever_rejects_non_list():
    with pytest.raises(AssertionError):
        HotpotQARetriever("not a list")


# ---------------------------------------------------------------------------
# HotpotQAEvaluator._download_datasets
# ---------------------------------------------------------------------------


def test_download_datasets_returns_path(tmp_path):
    """_download_datasets returns a dict mapping dataset name to file path."""
    evaluator = HotpotQAEvaluator()

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"[]"]

    with (
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.get_cache_dir",
            return_value=str(tmp_path),
        ),
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.requests.get",
            return_value=mock_response,
        ),
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.tqdm.tqdm",
            side_effect=lambda it, **kw: it,
        ),
    ):
        paths = evaluator._download_datasets()

    assert "hotpot_dev_distractor" in paths
    assert paths["hotpot_dev_distractor"].endswith("dev_distractor.json")


def test_download_datasets_raises_on_network_error(tmp_path):
    """_download_datasets raises ValueError when download fails."""
    evaluator = HotpotQAEvaluator()

    with (
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.get_cache_dir",
            return_value=str(tmp_path),
        ),
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.requests.get",
            side_effect=ConnectionError("network down"),
        ),
    ):
        with pytest.raises(ValueError, match="could not download"):
            evaluator._download_datasets()


def test_download_datasets_skips_download_if_file_exists(tmp_path):
    """If the dataset dir already exists, no network request is made."""
    dataset_dir = tmp_path / "datasets" / "HotpotQA"
    dataset_dir.mkdir(parents=True)
    dataset_file = dataset_dir / "dev_distractor.json"
    dataset_file.write_text("[]")

    evaluator = HotpotQAEvaluator()

    with (
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.get_cache_dir",
            return_value=str(tmp_path),
        ),
        patch(
            "llama_index.core.evaluation.benchmarks.hotpotqa.requests.get"
        ) as mock_get,
    ):
        paths = evaluator._download_datasets()
        mock_get.assert_not_called()

    assert "hotpot_dev_distractor" in paths


# ---------------------------------------------------------------------------
# HotpotQAEvaluator.run
# ---------------------------------------------------------------------------


def test_run_computes_scores(tmp_path):
    """run() queries the engine and completes without errors."""
    evaluator = HotpotQAEvaluator()

    dataset_file = tmp_path / "dev_distractor.json"
    dataset_file.write_text(json.dumps(SAMPLE_QUERY_OBJECTS))

    mock_query_engine = MagicMock(spec=RetrieverQueryEngine)
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Shakespeare"
    mock_query_engine.query.return_value = mock_response
    mock_query_engine.with_retriever.return_value = mock_query_engine

    with patch.object(
        evaluator,
        "_download_datasets",
        return_value={"hotpot_dev_distractor": str(dataset_file)},
    ):
        evaluator.run(mock_query_engine, queries=1)

    mock_query_engine.query.assert_called_once()
