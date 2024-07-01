import os

from llama_index.readers.box.box_client_ccg import (
    BoxConfigCCG,
    get_ccg_enterprise_client,
    get_ccg_user_client,
)


def test_box_load_env_credentials():
    box_config = BoxConfigCCG()
    assert box_config.client_id is not None
    assert box_config.client_secret is not None

    assert box_config.client_id == os.getenv("BOX_CLIENT_ID")
    assert box_config.client_secret == os.getenv("BOX_CLIENT_SECRET")


def test_login_service_account():
    box_config = BoxConfigCCG()
    client = get_ccg_enterprise_client(box_config)

    me = client.users.get_user_me()

    assert me is not None
    assert me.id is not None


def test_login_user_account():
    box_config = BoxConfigCCG()
    client = get_ccg_user_client(box_config, box_config.ccg_user_id)

    me = client.users.get_user_me()

    assert me is not None
    assert me.id is not None
