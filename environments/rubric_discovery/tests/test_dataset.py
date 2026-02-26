"""Tests for env/dataset/: loader and transform."""

import json
import os
import tempfile

import pytest

from environments.rubric_discovery.env.dataset.loader import (
    get_default_dataset_path,
    load_dataset,
    save_dataset,
)
from environments.rubric_discovery.env.dataset.transform import (
    prepare_episodes,
    transform_row,
)
from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.types import (
    DatasetRow,
    LabeledExample,
    RubricDiscoveryConfig,
)


@pytest.fixture
def sample_rows() -> list:
    return [
        DatasetRow(
            train_examples=[
                LabeledExample("q1", "a1", 0.5),
                LabeledExample("q2", "a2", 1.0),
            ],
            test_examples=[
                LabeledExample("q3", "a3", 0.7),
            ],
            category="test_cat",
        ),
        DatasetRow(
            train_examples=[
                LabeledExample("q4", "a4", 0.0),
            ],
            test_examples=[
                LabeledExample("q5", "a5", 1.0),
            ],
            category="other_cat",
        ),
    ]


class TestLoader:
    def test_load_default_dataset(self) -> None:
        path = get_default_dataset_path()
        assert os.path.exists(path), f"Default dataset not found: {path}"
        rows = load_dataset(path)
        assert len(rows) > 0

    def test_save_and_load(self, sample_rows: list) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            tmppath = f.name

        try:
            save_dataset(sample_rows, tmppath)
            loaded = load_dataset(tmppath)
            assert len(loaded) == 2
            assert loaded[0].category == "test_cat"
            assert len(loaded[0].train_examples) == 2
        finally:
            os.unlink(tmppath)

    def test_category_filter(self, sample_rows: list) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            tmppath = f.name

        try:
            save_dataset(sample_rows, tmppath)
            loaded = load_dataset(tmppath, categories=["test_cat"])
            assert len(loaded) == 1
            assert loaded[0].category == "test_cat"
        finally:
            os.unlink(tmppath)

    def test_max_examples(self, sample_rows: list) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            tmppath = f.name

        try:
            save_dataset(sample_rows, tmppath)
            loaded = load_dataset(tmppath, max_examples=1)
            assert len(loaded) == 1
        finally:
            os.unlink(tmppath)


class TestTransform:
    def test_transform_row(self) -> None:
        config = RubricDiscoveryConfig(eval_backend="subprocess")
        row = DatasetRow(
            train_examples=[LabeledExample("q", "a", 0.5)],
            test_examples=[LabeledExample("qt", "at", 1.0)],
            category="cat",
        )
        service = EvaluationService(config)
        episode = transform_row(row, config, service)

        assert "system_prompt" in episode
        assert "task_prompt" in episode
        assert "rubric_scorer" in episode
        assert "train_examples" in episode
        assert "test_examples" in episode
        assert episode["category"] == "cat"

    def test_prepare_episodes(self) -> None:
        config = RubricDiscoveryConfig(eval_backend="subprocess")
        rows = [
            DatasetRow(
                train_examples=[LabeledExample("q1", "a1", 0.5)],
                test_examples=[LabeledExample("qt1", "at1", 1.0)],
            ),
            DatasetRow(
                train_examples=[LabeledExample("q2", "a2", 0.5)],
                test_examples=[LabeledExample("qt2", "at2", 1.0)],
            ),
        ]
        service = EvaluationService(config)
        episodes = prepare_episodes(rows, config, service)
        assert len(episodes) == 2
