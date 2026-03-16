import json

from llama_index.core.schema import TextNode

from llama_index.vector_stores.bigquery import BigQueryVectorStore


def test_add_sends_correct_node_data_to_bigquery(vector_store: BigQueryVectorStore):
    """It should insert nodes into BigQuery and return the corresponding IDs."""
    # Given a list of nodes
    nodes = [
        TextNode(
            id_="node-id",
            text="Lorem Ipsum",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "unit-test"},
        )
    ]

    # When the nodes are added to the vector store
    result = vector_store.add(nodes)

    # Then the load job should be triggered
    vector_store.client.load_table_from_json.assert_called_once()

    # And the correct node data should be sent to BigQuery
    _, kwargs = vector_store.client.load_table_from_json.call_args
    json_rows = kwargs["json_rows"]
    expected_metadata = {
        "source": "unit-test",
        "_node_content": json.dumps(
            {
                "id_": "node-id",
                "embedding": None,
                "metadata": {"source": "unit-test"},
                "excluded_embed_metadata_keys": [],
                "excluded_llm_metadata_keys": [],
                "relationships": {},
                "metadata_template": "{key}: {value}",
                "metadata_separator": "\n",
                "text": "",
                "mimetype": "text/plain",
                "start_char_idx": None,
                "end_char_idx": None,
                "metadata_seperator": "\n",
                "text_template": "{metadata_str}\n\n{content}",
                "class_name": "TextNode",
            }
        ),
        "_node_type": "TextNode",
        "document_id": "None",
        "doc_id": "None",
        "ref_doc_id": "None",
    }
    assert json_rows[0]["node_id"] == "node-id"
    assert json_rows[0]["text"] == "Lorem Ipsum"
    assert json_rows[0]["embedding"] == [0.1, 0.2, 0.3]
    assert json_rows[0]["metadata"] == expected_metadata

    # And the returned node ID list should match the inserted node IDs
    assert result == ["node-id"]
