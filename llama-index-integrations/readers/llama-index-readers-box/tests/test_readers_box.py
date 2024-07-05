import pytest
import os
import dotenv
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReader

from box_sdk_gen import CCGConfig


@pytest.fixture(scope="module")
def box_environment_ccg():
    dotenv.load_dotenv()

    # Common configurations
    client_id = os.getenv("BOX_CLIENT_ID", "YOUR_BOX_CLIENT_ID")
    client_secret = os.getenv("BOX_CLIENT_SECRET", "YOUR_BOX_CLIENT_SECRET")

    # CCG configurations
    enterprise_id = os.getenv("BOX_ENTERPRISE_ID", "YOUR_BOX_ENTERPRISE_ID")
    ccg_user_id = os.getenv("BOX_CCG_USER_ID")

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "enterprise_id": enterprise_id,
        "ccg_user_id": ccg_user_id,
    }


@pytest.fixture(scope="module")
def box_unit_testing_config(box_environment_ccg: CCGConfig):
    box_config = box_environment_ccg
    return CCGConfig(
        client_id=box_config["client_id"],
        client_secret=box_config["client_secret"],
        enterprise_id=box_config["enterprise_id"],
        user_id=box_config["ccg_user_id"],
    )


@pytest.fixture(scope="module")
def box_integration_testing_config(box_unit_testing_config: CCGConfig):
    box_config = box_unit_testing_config
    if box_config.client_id == "YOUR_BOX_CLIENT_ID":
        raise pytest.skip(
            f"Create a .env file with the Box credentials to run integration tests."
        )
    return box_config


@pytest.fixture(scope="module")
def box_reader(box_integration_testing_config: CCGConfig):
    box_config = box_integration_testing_config

    return BoxReader(box_config=box_config)


def get_testing_data() -> dict:
    return {
        "disable_slow_tests": True,
        "test_folder_id": "273257908044",
        "test_doc_id": "1579334243393",
        "test_ppt_id": "994852771390",
        "test_xls_id": "994854421385",
        "test_pdf_id": "994851508870",
        "test_json_id": "1579338585099",
        "test_csv_id": "1579338385706",
    }


def test_class():
    names_of_base_classes = [b.__name__ for b in BoxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_unit_testing_config: CCGConfig):
    box_config = box_unit_testing_config
    reader = BoxReader(box_config=box_config)

    # schema = reader.schema()
    # assert schema is not None
    # assert len(schema) > 0
    # assert "box_client_id" in schema["properties"]

    # json = reader.json(exclude_unset=True)

    # new_reader = BoxReader.parse_raw(json)
    # assert new_reader is not None
    # assert new_reader.box_client_id == reader.box_client_id
    # assert new_reader.box_client_secret == reader.box_client_secret


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_csv(box_reader: BoxReader):
    test_data = get_testing_data()
    docs = box_reader.load_data(file_ids=[test_data["test_csv_id"]])
    assert len(docs) == 1


def test_box_reader_folder(box_reader: BoxReader):
    # Very slow test
    test_data = get_testing_data()
    if test_data["disable_slow_tests"]:
        raise pytest.skip(f"Slow integration tests are disabled.")

    docs = box_reader.load_data(folder_id=test_data["test_folder_id"])
    assert len(docs) >= 1
