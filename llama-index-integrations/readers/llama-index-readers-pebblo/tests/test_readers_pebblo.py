import pytest
import os

from llama_index.core.readers.base import BaseReader
from llama_index.readers.pebblo import PebbloSafeReader
from pathlib import Path
from typing import Dict
from pytest_mock import MockerFixture
from llama_index.readers.file import CSVReader


csv_empty_file_name = "test_empty.csv"
csv_file_name = "test_nominal.csv"


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int):
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


@pytest.fixture()
def create_empty_file():
    with open(csv_empty_file_name, "w"):
        pass

    yield
    if os.path.exists(csv_empty_file_name):
        os.remove(csv_empty_file_name)


@pytest.fixture()
def create_csv_file():
    data = "column1,column2,column3\nvalue1,value2,value3\nvalue4,value5,value6\n"
    with open(csv_file_name, "w") as csv_file:
        csv_file.write(data)

    yield
    if os.path.exists(csv_file_name):
        os.remove(csv_file_name)


def test_class():
    names_of_base_classes = [b.__name__ for b in PebbloSafeReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_empty_filebased_loader(mocker: MockerFixture, create_empty_file) -> None:
    """Test basic file based csv loader."""
    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )

    file_path = f"{Path().resolve()}/{csv_empty_file_name}"

    # Exercise
    loader = PebbloSafeReader(
        CSVReader(),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load_data(file=Path(file_path))

    # Assert
    assert result[0].text == ""
    assert result[0].metadata == {"filename": "test_empty.csv", "extension": ".csv"}


def test_csv_loader_load_valid_data(mocker: MockerFixture, create_csv_file) -> None:
    mocker.patch.multiple(
        "requests",
        get=MockResponse(json_data={"data": ""}, status_code=200),
        post=MockResponse(json_data={"data": ""}, status_code=200),
    )
    file_path = f"{Path().resolve()}/test_nominal.csv"

    # Exercise
    loader = PebbloSafeReader(
        CSVReader(),
        "dummy_app_name",
        "dummy_owner",
        "dummy_description",
    )
    result = loader.load_data(file=Path(file_path))

    # Assert
    assert (
        result[0].text
        == "column1, column2, column3\nvalue1, value2, value3\nvalue4, value5, value6"
    )
    assert result[0].metadata == {"filename": "test_nominal.csv", "extension": ".csv"}
