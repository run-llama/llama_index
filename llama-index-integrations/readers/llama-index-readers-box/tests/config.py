import pytest
from llama_index.readers.box.box_client_ccg import (
    BoxConfigCCG,
    get_ccg_enterprise_client,
)


@pytest.fixture(scope="module")
def box_client():
    box_config = BoxConfigCCG()
    return get_ccg_enterprise_client(box_config)


@pytest.fixture(scope="module")
def box_config():
    return BoxConfigCCG()
