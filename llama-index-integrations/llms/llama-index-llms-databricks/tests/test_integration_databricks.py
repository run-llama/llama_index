import os

import pytest

from llama_index.llms.databricks import DataBricks


@pytest.mark.skipif(
    "DATABRICKS_API_KEY" not in os.environ or "DATABRICKS_API_BASE" not in os.environ,
    reason="DATABRICKS_API_KEY or DATABRICKS_API_BASE not set in environment",
)
def test_completion():
    databricks = DataBricks(
        model="databricks-dbrx-instruct", temperature=0, max_tokens=2
    )
    resp = databricks.complete("hello")
    assert resp.text == "Hello"


@pytest.mark.skipif(
    "DATABRICKS_API_KEY" not in os.environ or "DATABRICKS_API_BASE" not in os.environ,
    reason="DATABRICKS_API_KEY or DATABRICKS_API_BASE not set in environment",
)
def test_stream_completion():
    databricks = DataBricks(
        model="databricks-dbrx-instruct", temperature=0, max_tokens=2
    )
    stream = databricks.stream_complete("hello")
    text = None
    for chunk in stream:
        text = chunk.text
    assert text == "Hello"
