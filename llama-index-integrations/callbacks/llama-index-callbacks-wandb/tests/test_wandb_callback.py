from llama_index.callbacks.wandb.base import WandbCallbackHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from unittest.mock import MagicMock, patch
import os
from pathlib import Path


def test_handler_callable():
    names_of_base_classes = [b.__name__ for b in WandbCallbackHandler.__mro__]
    assert BaseCallbackHandler.__name__ in names_of_base_classes


def test_persist_index_coverage(tmp_path: Path):
    """Test persist_index to cover initialization logic of _default_persist_dir."""
    mock_wandb = MagicMock()
    # Mocking the wandb.run.dir
    mock_wandb.run.dir = str(tmp_path / "wandb_test")

    # Mock wandb and trace_tree to avoid environment issues in CI
    with patch.dict(
        "sys.modules", {"wandb": mock_wandb, "wandb.sdk.data_types": MagicMock()}
    ):
        handler = WandbCallbackHandler(run_args={"project": "test"})
        # Override the internal wandb object with our mock
        handler._wandb = mock_wandb

        mock_index = MagicMock()

        test_dir = str(tmp_path / "test_persist_dir")

        # Calling with a custom persist_dir should trigger _default_persist_dir = False
        handler.persist_index(mock_index, "test_index", persist_dir=test_dir)
        assert os.path.exists(test_dir)
