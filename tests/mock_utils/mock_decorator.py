"""Shared decorator."""
import functools
from typing import Any, Callable
from unittest.mock import patch

from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from tests.mock_utils.mock_predict import mock_llmpredictor_predict
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


def patch_common(f: Callable) -> Callable:
    """Create patch decorator with common mocks."""

    @patch.object(
        TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline
    )
    @patch.object(LLMPredictor, "total_tokens_used", return_value=0)
    @patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
    @patch.object(LLMPredictor, "__init__", return_value=None)
    @functools.wraps(f)
    def functor(*args: Any, **kwargs: Any) -> Any:
        return f(*args, **kwargs)

    return functor
