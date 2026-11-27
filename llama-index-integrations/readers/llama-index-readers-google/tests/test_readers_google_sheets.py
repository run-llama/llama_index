from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from llama_index.readers.google import GoogleSheetsReader


def _service(sheets, values_by_range):
    """
    A ``discovery.build`` stand-in.

    ``spreadsheets().get`` answers ``sheets``, and ``values().get`` answers by the ``range`` it
    is asked for, recording every range sent.
    """
    service = MagicMock()
    service.spreadsheets().get().execute.return_value = {
        "sheets": [
            {
                "properties": {
                    "title": t,
                    "gridProperties": {"rowCount": r, "columnCount": c},
                }
            }
            for t, r, c in sheets
        ]
    }
    sent = []

    def values_get(spreadsheetId, range):
        sent.append(range)
        response = MagicMock()
        response.execute.return_value = {"values": values_by_range[range]}
        return response

    service.spreadsheets().values().get.side_effect = values_get
    return service, sent


@pytest.fixture
def reader():
    reader = GoogleSheetsReader()
    reader._get_credentials = MagicMock(return_value=MagicMock())
    return reader


def test_load_data_reads_each_sheet_under_its_own_title(reader):
    service, sent = _service(
        [("Revenue", 3, 2), ("Notes", 1, 1)],
        {
            "'Revenue'!R1C1:R3C2": [
                ["month", "revenue"],
                ["Jan", "120000"],
                ["Feb", "135000"],
            ],
            "'Notes'!R1C1:R1C1": [["Q1 closes 2026-03-31"]],
        },
    )
    with patch(
        "llama_index.readers.google.sheets.base.discovery.build", return_value=service
    ):
        docs = reader.load_data(spreadsheet_ids=["sheet-id"])

    assert sent == ["'Revenue'!R1C1:R3C2", "'Notes'!R1C1:R1C1"]
    assert len(docs) == 1
    assert docs[0].text == (
        "Revenue\nmonth\trevenue\nJan\t120000\nFeb\t135000\nNotes\nQ1 closes 2026-03-31\n"
    )


def test_a_title_that_needs_quoting_is_quoted(reader):
    service, sent = _service(
        [("Q1 Plan", 1, 1), ("A1", 1, 1), ("it's", 1, 1)],
        {
            "'Q1 Plan'!R1C1:R1C1": [["a"]],
            "'A1'!R1C1:R1C1": [["b"]],
            "'it''s'!R1C1:R1C1": [["c"]],
        },
    )
    with patch(
        "llama_index.readers.google.sheets.base.discovery.build", return_value=service
    ):
        docs = reader.load_data(spreadsheet_ids=["sheet-id"])

    assert sent == ["'Q1 Plan'!R1C1:R1C1", "'A1'!R1C1:R1C1", "'it''s'!R1C1:R1C1"]
    assert docs[0].text == "Q1 Plan\na\nA1\nb\nit's\nc\n"


def test_load_data_in_pandas_sends_the_same_ranges(reader):
    service, sent = _service(
        [("Revenue", 3, 2), ("Notes", 1, 1)],
        {
            "'Revenue'!R1C1:R3C2": [
                ["month", "revenue"],
                ["Jan", "120000"],
                ["Feb", "135000"],
            ],
            "'Notes'!R1C1:R1C1": [["Q1 closes 2026-03-31"]],
        },
    )
    with patch(
        "llama_index.readers.google.sheets.base.discovery.build", return_value=service
    ):
        frames = reader.load_data_in_pandas(spreadsheet_ids=["sheet-id"])

    assert sent == ["'Revenue'!R1C1:R3C2", "'Notes'!R1C1:R1C1"]
    assert [f.shape for f in frames] == [(2, 2), (0, 1)]
    assert list(frames[0].columns) == ["month", "revenue"]
    assert isinstance(frames[1], pd.DataFrame)
