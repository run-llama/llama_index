"""Shared decorator."""
from unittest.mock import patch

from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.langchain_helpers.chain_wrapper import LLMChain, LLMPredictor
from tests.mock_utils.mock_predict import (
    mock_llmchain_predict,
    mock_llmpredictor_predict,
)
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline
import functools

def patch_common(f):
    @patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
    @patch.object(LLMPredictor, "total_tokens_used", return_value=0)
    @patch.object(LLMPredictor, "predict", side_effect=mock_llmpredictor_predict)
    @patch.object(LLMPredictor, "__init__", return_value=None)
    @functools.wraps(f)
    def functor(*args, **kwargs):
        return f(*args, **kwargs)
    return functor
