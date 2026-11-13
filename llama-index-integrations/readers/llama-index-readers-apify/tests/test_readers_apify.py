from datetime import timedelta
from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.apify import ApifyActor, ApifyDataset

ITEMS = [
    {"text": "first page", "url": "https://example.com/a"},
    {"text": "second page", "url": "https://example.com/b"},
]


def _to_document(item: Dict) -> Document:
    return Document(text=item["text"], metadata={"url": item["url"]})


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in ApifyDataset.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ApifyActor.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@patch("apify_client.ApifyClient")
def test_dataset_load_data(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    client.dataset.return_value.list_items.return_value = SimpleNamespace(items=ITEMS)

    documents = ApifyDataset("token").load_data(
        dataset_id="dataset-id", dataset_mapping_function=_to_document
    )

    mock_client_cls.assert_called_once_with(
        "token", user_agent_suffix="Origin/llama_index"
    )
    client.dataset.assert_called_once_with("dataset-id")
    client.dataset.return_value.list_items.assert_called_once_with(clean=True)
    assert [d.text for d in documents] == ["first page", "second page"]
    assert documents[0].metadata == {"url": "https://example.com/a"}


@patch("apify_client.ApifyClient")
def test_dataset_mapping_must_return_document(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    client.dataset.return_value.list_items.return_value = SimpleNamespace(items=ITEMS)

    with pytest.raises(ValueError, match="must return a Document"):
        ApifyDataset("token").load_data(
            dataset_id="dataset-id", dataset_mapping_function=lambda item: item
        )


@patch("apify_client.ApifyClient")
def test_actor_load_data(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    # apify-client 3 returns a Run model: the dataset ID is an attribute, not a dict key.
    client.actor.return_value.call.return_value = SimpleNamespace(
        default_dataset_id="run-dataset-id"
    )
    client.dataset.return_value.list_items.return_value = SimpleNamespace(items=ITEMS)

    documents = ApifyActor("token").load_data(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": "https://example.com"}]},
        dataset_mapping_function=_to_document,
        memory_mbytes=1024,
        timeout_secs=60,
    )

    client.actor.assert_called_once_with("apify/website-content-crawler")
    client.actor.return_value.call.assert_called_once_with(
        run_input={"startUrls": [{"url": "https://example.com"}]},
        build=None,
        memory_mbytes=1024,
        run_timeout=timedelta(seconds=60),
    )
    client.dataset.assert_called_once_with("run-dataset-id")
    assert len(documents) == 2


@patch("apify_client.ApifyClient")
def test_actor_load_data_without_timeout(mock_client_cls: MagicMock) -> None:
    client = mock_client_cls.return_value
    client.actor.return_value.call.return_value = SimpleNamespace(
        default_dataset_id="run-dataset-id"
    )
    client.dataset.return_value.list_items.return_value = SimpleNamespace(items=[])

    ApifyActor("token").load_data(
        actor_id="user/actor", run_input={}, dataset_mapping_function=_to_document
    )

    assert client.actor.return_value.call.call_args.kwargs["run_timeout"] is None


@patch("apify_client.ApifyClient")
def test_actor_load_data_failed_call(mock_client_cls: MagicMock) -> None:
    mock_client_cls.return_value.actor.return_value.call.return_value = None

    with pytest.raises(RuntimeError, match="Failed to call Actor user/actor"):
        ApifyActor("token").load_data(
            actor_id="user/actor", run_input={}, dataset_mapping_function=_to_document
        )
