import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "llama_index.core.evaluation",
        "llama_index.core.retrievers",
        "llama_index.core.storage.kvstore",
    ],
)
def test_all_names_resolve(module_name: str) -> None:
    """Every name in ``__all__`` must exist, otherwise ``import *`` raises."""
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []

    # the user-facing symptom
    exec(f"from {module_name} import *", {})
