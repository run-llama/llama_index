import os
import dotenv
import pytest
from box_sdk_gen import CCGConfig, BoxCCGAuth, BoxClient, JWTConfig, BoxJWTAuth


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


@pytest.fixture(scope="module")
def box_client_ccg_integration_testing(box_environment_ccg):
    config = CCGConfig(
        client_id=box_environment_ccg["client_id"],
        client_secret=box_environment_ccg["client_secret"],
        enterprise_id=box_environment_ccg["enterprise_id"],
        user_id=box_environment_ccg["ccg_user_id"],
    )
    if config.client_id == "YOUR_BOX_CLIENT_ID":
        raise pytest.skip(
            f"Create a .env file with the Box credentials to run integration tests."
        )
    auth = BoxCCGAuth(config)
    if config.user_id:
        auth.with_user_subject(config.user_id)
    return BoxClient(auth)


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
def box_client_jwt_unit_testing(box_environment_jwt):
    # check if .env file is configured
    jwt_config_path = box_environment_jwt["jwt_config_path"]
    if not os.path.exists(jwt_config_path):
        config = JWTConfig(
            client_id="YOUR_BOX_CLIENT_ID",
            client_secret="YOUR_BOX_CLIENT_SECRET",
            jwt_key_id="YOUR_BOX_JWT_KEY_ID",
            private_key="YOUR_BOX_PRIVATE_KEY",
            private_key_passphrase="YOUR_BOX_PRIVATE_KEY_PASSPHRASE",
            enterprise_id="YOUR_BOX_ENTERPRISE_ID",
        )
    else:
        config = JWTConfig.from_config_file(jwt_config_path)
        user_id = box_environment_jwt["jwt_user_id"]
        if user_id:
            config.user_id = user_id
            config.enterprise_id = None
    auth = BoxJWTAuth(config)
    return BoxClient(auth)


@pytest.fixture(scope="module")
def box_client_jwt_integration_testing(box_environment_jwt):
    jwt_config_path = box_environment_jwt["jwt_config_path"]
    if not os.path.exists(jwt_config_path):
        config = JWTConfig(
            client_id="YOUR_BOX_CLIENT_ID",
            client_secret="YOUR_BOX_CLIENT_SECRET",
            jwt_key_id="YOUR_BOX_JWT_KEY_ID",
            private_key="YOUR_BOX_PRIVATE_KEY",
            private_key_passphrase="YOUR_BOX_PRIVATE_KEY_PASSPHRASE",
        )
    else:
        config = JWTConfig.from_config_file(jwt_config_path)
        user_id = box_environment_jwt["jwt_user_id"]
        if user_id:
            config.user_id = user_id
            config.enterprise_id = None

    if config.client_id == "YOUR_BOX_CLIENT_ID":
        raise pytest.skip(
            f"Create a .env file with the Box credentials to run integration tests."
        )
    auth = BoxJWTAuth(config)
    return BoxClient(auth)


def get_testing_data() -> dict:
    return {
        "disable_folder_tests": True,
        "test_folder_id": "273980493541",
        "test_doc_id": "1584054722303",
        "test_ppt_id": "1584056661506",
        "test_xls_id": "1584048916472",
        "test_pdf_id": "1584049890463",
        "test_json_id": "1584058432468",
        "test_csv_id": "1584054196674",
        "test_txt_waiver_id": "1514587167701",
        "test_folder_invoice_po_id": "261452450320",
        "test_txt_invoice_id": "1517629086517",
        "test_txt_po_id": "1517628697289",
        "metadata_template_key": "rbInvoicePO",
        "metadata_enterprise_scope": "enterprise_" + os.getenv("BOX_ENTERPRISE_ID"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
    }
