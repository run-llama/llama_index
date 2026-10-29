from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from llama_index.readers.file.tabular import PandasExcelReader


@pytest.mark.parametrize("sheet_name", [None, "Sheet"])
@pytest.mark.parametrize("concat_rows", [True, False])
def test_excel_preserves_integer_cells(
    tmp_path: Path, sheet_name: str | None, concat_rows: bool
) -> None:
    file = tmp_path / "identifiers.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["identifier", "price"])
    # A text cell preserves all digits in the workbook before pandas infers int64.
    sheet.append(["9007199254740993", 1.5])
    sheet.append(["42", 2.5])
    workbook.save(file)
    workbook.close()
    assert pd.read_excel(file)["identifier"].tolist() == [9007199254740993, 42]

    documents = PandasExcelReader(
        sheet_name=sheet_name, concat_rows=concat_rows
    ).load_data(file, extra_info={"source": "identifiers"})

    rows = ["identifier: 9007199254740993, price: 1.5", "identifier: 42, price: 2.5"]
    assert [document.text for document in documents] == (
        ["\n".join(rows)] if concat_rows else rows
    )
    assert all(document.metadata == {"source": "identifiers"} for document in documents)


@pytest.mark.parametrize("sheet_name", [None, "Sheet"])
def test_excel_preserves_missing_cells_and_custom_separators(
    tmp_path: Path, sheet_name: str | None
) -> None:
    file = tmp_path / "mixed.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["name", "count"])
    sheet.append(["Ada", None])
    sheet.append(["Grace", 2])
    workbook.save(file)
    workbook.close()

    documents = PandasExcelReader(
        sheet_name=sheet_name, field_separator=" | ", key_value_separator="="
    ).load_data(file)

    assert documents[0].text == "name=Ada | count=\nname=Grace | count=2.0"
