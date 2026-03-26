from datetime import date, datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc
from typing import Any, Union
from zoneinfo import ZoneInfo

import pytest

from llama_index.vector_stores.solr.client.utils import (
    format_datetime_for_solr,
    prepare_document_for_solr,
)


@pytest.mark.parametrize(
    ("input_dt", "expected_output"),
    [
        (datetime(2025, 2, 18, 1, 2, 3, tzinfo=UTC), "2025-02-18T01:02:03Z"),
        (datetime(2025, 2, 18, 1, 2, 3), "2025-02-18T01:02:03Z"),
        (
            datetime(2025, 2, 18, 1, 2, 3, tzinfo=ZoneInfo("America/New_York")),
            "2025-02-18T06:02:03Z",
        ),
        (date(2025, 2, 18), "2025-02-18T00:00:00Z"),
    ],
    ids=["UTC datetime", "Naive datetime", "Local datetime", "Date (no timezone)"],
)
def test_format_datetime_for_solr(
    input_dt: Union[datetime, date], expected_output: str
) -> None:
    # WHEN
    actual_output = format_datetime_for_solr(input_dt)

    # THEN
    assert actual_output == expected_output


def test_prepare_document_for_solr(
    mock_solr_raw_input_documents: list[dict[str, Any]],
    mock_solr_updated_input_documents: list[dict[str, Any]],
) -> None:
    # WHEN
    actual_updated_docs = [
        prepare_document_for_solr(doc) for doc in mock_solr_raw_input_documents
    ]

    # THEN
    assert actual_updated_docs == mock_solr_updated_input_documents
