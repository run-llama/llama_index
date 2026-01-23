import pytest
from unittest.mock import MagicMock, patch
from typing import Iterable

from datasets import Dataset

from llama_index.readers.datasets import DatasetsReader


@pytest.fixture
def reader():
    return DatasetsReader()


@pytest.fixture
def sample_data():
    return [
        {"id": "doc_1", "content": "This is the first document.", "extra": "A"},
        {"id": "doc_2", "content": "This is the second document.", "extra": "B"},
    ]


# --- load_data tests ---


def test_load_data_with_preloaded_dataset(reader, sample_data):
    """Test load_data with a preloaded dataset."""
    # Mocking a Dataset object that behaves like a list
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__iter__.return_value = iter(sample_data)

    docs = reader.load_data(dataset=mock_dataset, text_key="content")

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert docs[0].text == "This is the first document."
    assert docs[0].metadata == sample_data[0]
    assert docs[1].text == "This is the second document."


@patch("llama_index.readers.datasets.base.load_dataset")
def test_load_data_from_huggingface(mock_hf_load, reader, sample_data):
    """Test load_data with a huggingface dataset."""
    # Setup the mock to return our sample data when iterated
    mock_ds_instance = MagicMock(spec=Dataset)
    mock_ds_instance.__iter__.return_value = iter(sample_data)
    mock_hf_load.return_value = mock_ds_instance

    dataset_name = "some/dataset"
    split_name = "validation"

    docs = reader.load_data(
        dataset_name, split=split_name, text_key="content", doc_id_key="id"
    )

    assert len(docs) == 2
    assert docs[0].id_ == "doc_1"
    assert docs[0].text == "This is the first document."

    mock_hf_load.assert_called_once_with(
        dataset_name, split=split_name, streaming=False
    )


# --- lazy_load_data tests ---


def test_lazy_load_data_with_preloaded_dataset(reader, sample_data):
    """Test lazy_load_data with a preloaded dataset."""
    # IterableDataset is basically just an iterable generator
    mock_iterable_ds = (item for item in sample_data)

    doc_gen = reader.lazy_load_data(dataset=mock_iterable_ds, text_key="content")

    assert isinstance(doc_gen, Iterable)
    # Ensure it's not a list yet
    assert not isinstance(doc_gen, list)

    # Consume generator
    docs = list(doc_gen)
    assert len(docs) == 2
    assert docs[0].text == sample_data[0]["content"]


@patch("llama_index.readers.datasets.base.load_dataset")
def test_lazy_load_data_from_huggingface(mock_hf_load, reader, sample_data):
    """Test lazy_load_data with a huggingface dataset."""
    # Setup mock to return an iterable
    mock_hf_load.return_value = iter(sample_data)

    dataset_name = "some/streamed_dataset"

    doc_gen = reader.lazy_load_data(dataset_name, text_key="content")

    assert isinstance(doc_gen, Iterable)

    mock_hf_load.assert_called_once_with(dataset_name, split="train", streaming=True)

    first_doc = next(doc_gen)
    assert first_doc.text == "This is the first document."
