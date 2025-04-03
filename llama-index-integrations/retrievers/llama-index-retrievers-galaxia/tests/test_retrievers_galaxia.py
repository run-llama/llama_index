from unittest.mock import MagicMock
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.galaxia import GalaxiaRetriever


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
