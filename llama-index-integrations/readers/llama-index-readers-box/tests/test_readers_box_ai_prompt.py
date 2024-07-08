import pytest
import os
import dotenv
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReaderAIPrompt

from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient


@pytest.fixture(scope="module")
def box_environment_ccg():
    dotenv.load_dotenv()

    # Common configurations
    client_id = os.getenv("BOX_CLIENT_ID", "YOUR_BOX_CLIENT_ID")
    client_secret = os.getenv("BOX_CLIENT_SECRET", "YOUR_BOX_CLIENT_SECRET")

    # CCG configurations
    enterprise_id = os.getenv("BOX_ENTERPRISE_ID", "YOUR_BOX_ENTERPRISE_ID")
    ccg_user_id = os.getenv("BOX_USER_ID")

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "enterprise_id": enterprise_id,
        "ccg_user_id": ccg_user_id,
    }


@pytest.fixture(scope="module")
def box_client_ccg_unit_testing(box_environment_ccg):
    config = CCGConfig(
        client_id=box_environment_ccg["client_id"],
        client_secret=box_environment_ccg["client_secret"],
        enterprise_id=box_environment_ccg["enterprise_id"],
        user_id=box_environment_ccg["ccg_user_id"],
    )
    auth = BoxCCGAuth(config)
    if config.user_id:
        auth.with_user_subject(config.user_id)
    return BoxClient(auth)


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


def test_class_name():
    names_of_base_classes = [b.__name__ for b in BoxReaderAIPrompt.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(box_client_ccg_unit_testing: BoxClient):
    reader = BoxReaderAIPrompt(box_client=box_client_ccg_unit_testing)


####################################################################################################
# Integration tests
####################################################################################################
