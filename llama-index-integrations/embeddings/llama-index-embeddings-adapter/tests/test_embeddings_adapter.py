from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.adapter import AdapterEmbeddingModel
from unittest.mock import patch, MagicMock
import json


def test_class():
    names_of_base_classes = [b.__name__ for b in AdapterEmbeddingModel.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes

## Test 
def test_load_uses_weights_only(tmp_path):
    """Test that torch.load is called with weights_only=True for security."""
    # Create fake config file
    config = {"in_features": 4, "out_features": 4, "bias": False}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    # Create fake model file
    model_path = tmp_path / "pytorch_model.bin"
    model_path.write_bytes(b"fake")

    from llama_index.embeddings.adapter.utils import LinearLayer

    with patch("llama_index.embeddings.adapter.utils.torch.load") as mock_load:
        mock_load.return_value = LinearLayer(4, 4).state_dict()

        LinearLayer.load(str(tmp_path))

        # Verify weights_only=True was passed
        assert mock_load.called, "torch.load was never called!"
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get("weights_only") is True, \
            "Security issue: torch.load must use weights_only=True!"