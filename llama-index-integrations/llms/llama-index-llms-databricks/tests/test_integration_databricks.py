import os

import pytest

from llama_index.llms.databricks import Databricks


@pytest.mark.skipif(
    "DATABRICKS_TOKEN" not in os.environ
    or "DATABRICKS_SERVING_ENDPOINT" not in os.environ,
    reason="DATABRICKS_TOKEN or DATABRICKS_SERVING_ENDPOINT not set in environment",
)
def test_completion():
    databricks = Databricks(
        model="databricks-dbrx-instruct", temperature=0, max_tokens=2
    )
    resp = databricks.complete("hello")
    assert resp.text == "Hello"


@pytest.mark.skipif(
    "DATABRICKS_TOKEN" not in os.environ
    or "DATABRICKS_SERVING_ENDPOINT" not in os.environ,
    reason="DATABRICKS_TOKEN or DATABRICKS_SERVING_ENDPOINT not set in environment",
)
def test_stream_completion():
    databricks = Databricks(
        model="databricks-dbrx-instruct", temperature=0, max_tokens=2
    )
    stream = databricks.stream_complete("hello")
    text = None
    for chunk in stream:
        text = chunk.text
    assert text == "Hello"
