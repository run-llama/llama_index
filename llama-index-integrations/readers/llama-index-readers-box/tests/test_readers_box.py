import os

from llama_index.readers.box.box_client_ccg import BoxConfigCCG


def test_box_load_env_credentials():
    box_config = BoxConfigCCG()
    assert box_config.client_id is not None
    assert box_config.client_secret is not None

    assert box_config.client_id == os.getenv("BOX_CLIENT_ID")
    assert box_config.client_secret == os.getenv("BOX_CLIENT_SECRET")
