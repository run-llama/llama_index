from unittest import mock

import pytest
from llama_index.postprocessor.aimon_rerank import AIMonRerank


def test_missing_api_key_raises_value_error():
    """A missing AIMON_API_KEY must raise a clear ValueError, not a raw KeyError."""
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Must pass in AIMon API key"):
            AIMonRerank()
