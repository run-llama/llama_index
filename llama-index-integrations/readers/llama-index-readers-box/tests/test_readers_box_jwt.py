import pytest
import os
import dotenv
from llama_index.core.readers.base import BaseReader
from llama_index.readers.box import BoxReader

from box_sdk_gen import JWTConfig

from tests.config import get_testing_data


@pytest.fixture(scope="module")
def box_environment_jwt():
    dotenv.load_dotenv()

    # JWT configurations
    jwt_config_path = os.getenv("JWT_CONFIG_PATH", ".jwt.config.json")
    jwt_user_id = os.getenv("BOX_USER_ID")

    return {
        "jwt_config_path": jwt_config_path,  # Path to the JWT config file
        "jwt_user_id": jwt_user_id,
    }


@pytest.fixture(scope="module")
def jwt_unit_testing(box_environment_jwt):
    # check if .env file is configured
    jwt_config_path = box_environment_jwt["jwt_config_path"]
    if not os.path.exists(jwt_config_path):
        return JWTConfig(
            client_id="YOUR_BOX_CLIENT_ID",
            client_secret="YOUR_BOX_CLIENT_SECRET",
            jwt_key_id="YOUR_BOX_JWT_KEY_ID",
            private_key="YOUR_BOX_PRIVATE_KEY",
            private_key_passphrase="YOUR_BOX_PRIVATE_KEY_PASSPHRASE",
        )
    else:
        jwt = JWTConfig.from_config_file(jwt_config_path)
        user_id = box_environment_jwt["jwt_user_id"]
        if user_id:
            jwt.user_id = user_id
            jwt.enterprise_id = None
        return jwt


@pytest.fixture(scope="module")
def jwt_integration_testing(jwt_unit_testing: JWTConfig):
    if jwt_unit_testing.client_id == "YOUR_BOX_CLIENT_ID":
        raise pytest.skip(
            f"Create a .env file with the Box credentials to run integration tests."
        )
    return jwt_unit_testing


@pytest.fixture(scope="module")
def box_reader_jwt(jwt_integration_testing: JWTConfig):
    return BoxReader(box_config=jwt_integration_testing)


def test_class():
    names_of_base_classes = [b.__name__ for b in BoxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_reader_init(jwt_unit_testing: JWTConfig):
    reader = BoxReader(box_config=jwt_unit_testing)


####################################################################################################
# Integration tests
####################################################################################################


def test_box_reader_csv(box_reader_jwt: BoxReader):
    test_data = get_testing_data()
    docs = box_reader_jwt.load_data(file_ids=[test_data["test_csv_id"]])
    assert len(docs) == 1


def test_box_reader_folder(box_reader_jwt: BoxReader):
    # Very slow test
    test_data = get_testing_data()
    if test_data["disable_slow_tests"]:
        raise pytest.skip(f"Slow integration tests are disabled.")

    docs = box_reader_jwt.load_data(folder_id=test_data["test_folder_id"])
    assert len(docs) >= 1
