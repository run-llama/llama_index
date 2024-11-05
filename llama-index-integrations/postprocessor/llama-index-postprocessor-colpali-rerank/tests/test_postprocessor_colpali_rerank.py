from unittest.mock import patch
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.colpali_rerank import ColPaliRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in ColPaliRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes


@patch("llama_index.postprocessor.colpali_rerank.base.ColPali")
@patch("llama_index.postprocessor.colpali_rerank.base.ColPaliProcessor")
def test_init(mock_processor, mock_model):
    # Setup mock returns
    mock_model.from_pretrained.return_value = MagicMock()
    mock_processor.from_pretrained.return_value = MagicMock()

    m = ColPaliRerank(top_n=10)

    assert m.model == "vidore/colpali-v1.2"
    assert m.top_n == 10

    # Verify the model was initialized with correct parameters
    mock_model.from_pretrained.assert_called_once()
    mock_processor.from_pretrained.assert_called_once()
