import warnings

from fsspec.implementations.memory import MemoryFileSystem
import pandas as pd
import pytest

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file.tabular import PandasCSVReader

# NA is Namibia's country code, a ZIP code may start with 0, a price may keep
# its trailing zero, and a cell may be blank or say N/A or NULL.
SAMPLE_CSV = """country,code,zip,amount,status
Namibia,NA,02139,1.50,N/A
Germany,DE,10115,,active
Norway,NO,0150,3,NULL
"""

AS_WRITTEN = [
    "Namibia, NA, 02139, 1.50, N/A",
    "Germany, DE, 10115, , active",
    "Norway, NO, 0150, 3, NULL",
]


@pytest.fixture()
def csv_file(tmp_path):
    file = tmp_path / "countries.csv"
    file.write_text(SAMPLE_CSV)
    return file


def test_pandas_csv_reader_reads_cells_as_written(csv_file):
    documents = PandasCSVReader().load_data(csv_file)

    assert len(documents) == 1
    assert documents[0].text == "\n".join(AS_WRITTEN)


def test_simple_directory_reader_reads_csv_cells_as_written(csv_file):
    # PandasCSVReader is the default reader for .csv files.
    documents = SimpleDirectoryReader(input_files=[csv_file]).load_data()

    assert [document.text for document in documents] == ["\n".join(AS_WRITTEN)]


def test_pandas_csv_reader_reads_cells_as_written_per_row(csv_file):
    documents = PandasCSVReader(concat_rows=False).load_data(csv_file)

    assert [document.text for document in documents] == AS_WRITTEN


def test_pandas_csv_reader_reads_cells_as_written_from_a_filesystem():
    fs = MemoryFileSystem()
    fs.pipe("/data/countries.csv", SAMPLE_CSV.encode())

    documents = PandasCSVReader().load_data("/data/countries.csv", fs=fs)

    assert documents[0].text == "\n".join(AS_WRITTEN)


def test_pandas_csv_reader_leaves_the_cells_a_short_row_lacks_empty(tmp_path):
    file = tmp_path / "short.csv"
    file.write_text("a,b,c\n1,2,3\n4\n")

    documents = PandasCSVReader().load_data(file)

    assert documents[0].text == "1, 2, 3\n4, , "


def test_pandas_csv_reader_lets_the_caller_keep_pandas_na_handling(csv_file):
    reader = PandasCSVReader(pandas_config={"keep_default_na": True})

    documents = reader.load_data(csv_file)

    # The N/A cells are missing now, and are left empty rather than "nan".
    assert documents[0].text == (
        "Namibia, , 02139, 1.50, \nGermany, DE, 10115, , active\nNorway, NO, 0150, 3, "
    )


def test_pandas_csv_reader_passes_other_options_through(tmp_path):
    file = tmp_path / "semicolons.csv"
    file.write_text("name;code\nNamibia;NA\n")

    documents = PandasCSVReader(pandas_config={"sep": ";"}).load_data(file)

    assert documents[0].text == "Namibia, NA"


def test_pandas_csv_reader_writes_integers_pandas_parses_as_integers(tmp_path):
    file = tmp_path / "numbers.csv"
    file.write_text("count,price\n3,1.5\n4,2.25\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        documents = PandasCSVReader(pandas_config={"dtype": None}).load_data(file)

    assert documents[0].text == "3, 1.5\n4, 2.25"


@pytest.mark.parametrize(
    ("pandas_config", "expected"),
    [
        # Read as text first, a date would be written in nanoseconds.
        ({"parse_dates": ["day"]}, "2024-01-05 00:00:00, 1.5, DE\n, , "),
        # Read with blank cells kept as text, a float column fails to parse.
        ({"dtype": {"price": float}}, "2024-01-05, 1.5, DE\n, , "),
        # Read as text, pandas warns that the converter overrides the dtype.
        ({"converters": {"code": str.lower}}, "2024-01-05, 1.5, de\n, , na"),
    ],
)
def test_pandas_csv_reader_leaves_conversions_the_caller_asks_for_to_pandas(
    tmp_path, pandas_config, expected
):
    file = tmp_path / "prices.csv"
    file.write_text("day,price,code\n2024-01-05,1.5,DE\n,,NA\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        documents = PandasCSVReader(pandas_config=pandas_config).load_data(file)

    assert documents[0].text == expected
