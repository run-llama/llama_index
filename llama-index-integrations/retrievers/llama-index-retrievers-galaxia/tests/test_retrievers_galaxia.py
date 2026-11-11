import json
from unittest.mock import MagicMock, patch
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.galaxia import GalaxiaRetriever
from llama_index.retrievers.galaxia.base import GalaxiaClient


def _mock_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    return response


def test_retrieve():
    mock_client = MagicMock()
    mock_client.retrieve.return_value = [
        {
            "group": "test_file.txt",
            "text": "test question?",
            "model": "human_model_name",
            "category": "* retrieved text",
            "result": "* retrieved",
            "rank": 0.55,
            "confidence": "Medium",
            "cells": [],
        },
    ]

    gr = GalaxiaRetriever("", "", "")
    gr._client = mock_client

    question = "test"
    result = gr.retrieve(QueryBundle(question))

    expected_result = [
        NodeWithScore(
            node=TextNode(
                id_="fbfee559-068c-4017-bd87-835e6da12a47",
                embedding=None,
                metadata={
                    "model": "human_model_name",
                    "file": "test_file.txt",
                },
                excluded_embed_metadata_keys=[],
                excluded_llm_metadata_keys=[],
                relationships={},
                text="* retrieved text",
                mimetype="text/plain",
                start_char_idx=None,
                end_char_idx=None,
                text_template="{metadata_str}\n\n{content}",
                metadata_template="{key}: {value}",
                metadata_seperator="\n",
            ),
            score=0.55,
        )
    ]

    assert result[0].text == expected_result[0].text
    assert result[0].score == expected_result[0].score
    assert result[0].metadata == expected_result[0].metadata


@patch("llama_index.retrievers.galaxia.base.http.client.HTTPSConnection")
def test_galaxia_client_retrieve_success_closes_connection(mock_https_connection):
    mock_conn = MagicMock()
    mock_https_connection.return_value = mock_conn
    # first init attempt fails (no operationId, exercises the retry/sleep
    # branch), second attempt succeeds; status is processed on first check
    mock_conn.getresponse.side_effect = [
        _mock_response({}),
        _mock_response({"operationId": "op-1"}),
        _mock_response({"status": "processed"}),
        _mock_response({"result": {"resultItems": ["item-1"]}}),
    ]

    client = GalaxiaClient("api.example.com", "key", "kb-1", n_retries=2, wait_time=0)
    result = client.retrieve("question")

    assert result == ["item-1"]
    mock_conn.close.assert_called_once()


@patch("llama_index.retrievers.galaxia.base.http.client.HTTPSConnection")
def test_galaxia_client_retrieve_closes_connection_on_failed_init(
    mock_https_connection,
):
    mock_conn = MagicMock()
    mock_https_connection.return_value = mock_conn
    mock_conn.getresponse.return_value = _mock_response({})

    client = GalaxiaClient("api.example.com", "key", "kb-1", n_retries=1, wait_time=0)
    result = client.retrieve("question")

    assert result is None
    mock_conn.close.assert_called_once()


@patch("llama_index.retrievers.galaxia.base.http.client.HTTPSConnection")
def test_galaxia_client_retrieve_closes_connection_on_failed_processing(
    mock_https_connection,
):
    mock_conn = MagicMock()
    mock_https_connection.return_value = mock_conn
    mock_conn.getresponse.side_effect = [
        _mock_response({"operationId": "op-1"}),
        _mock_response({"status": "pending"}),
    ]

    client = GalaxiaClient("api.example.com", "key", "kb-1", n_retries=1, wait_time=0)
    result = client.retrieve("question")

    assert result is None
    mock_conn.close.assert_called_once()
