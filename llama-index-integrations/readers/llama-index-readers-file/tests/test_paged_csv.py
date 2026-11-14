from llama_index.readers.file import PagedCSVReader


def test_paged_csv_reader_one_document_per_row(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,city\nAda,London\nGrace,New York\n")

    docs = PagedCSVReader().load_data(path)

    assert [d.text for d in docs] == [
        "name: Ada\ncity: London",
        "name: Grace\ncity: New York",
    ]


def test_paged_csv_reader_row_with_missing_cells(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,city\nAda\nGrace,New York\n")

    docs = PagedCSVReader().load_data(path)

    assert [d.text for d in docs] == [
        "name: Ada\ncity: ",
        "name: Grace\ncity: New York",
    ]


def test_paged_csv_reader_row_with_extra_cells(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,city\nAda,London,UK\nGrace,New York\n")

    docs = PagedCSVReader().load_data(path)

    assert [d.text for d in docs] == [
        "name: Ada\ncity: London",
        "name: Grace\ncity: New York",
    ]


def test_paged_csv_reader_row_missing_several_cells(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b,c\n1\n")

    docs = PagedCSVReader().load_data(path)

    assert [d.text for d in docs] == ["a: 1\nb: \nc: "]


def test_paged_csv_reader_empty_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("")

    assert PagedCSVReader().load_data(path) == []


def test_paged_csv_reader_header_only(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,city\n")

    assert PagedCSVReader().load_data(path) == []
