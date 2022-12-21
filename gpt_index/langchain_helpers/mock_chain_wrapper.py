"""Mock chain wrapper."""

from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.base import Prompt
from typing import Any, Tuple, Dict
from gpt_index.constants import NUM_OUTPUTS

# TODO: consolidate with unit tests in tests/mock_utils/mock_predict.py

def _mock_answer(max_tokens: int) -> str:
    """Mock answer."""
    return ["mock"] * max_tokens

def _mock_refine(max_tokens: int) -> str:
    """Mock refine."""
    return ["mock"] * max_tokens

def _mock_keyword_extract(prompt_args: Dict) -> str:
    """Mock keyword extract."""
    max_keywords = prompt_args["max_keywords"]
    keywords_str = ["keyword"] * max_keywords
    return f"KEYWORDS: {keywords_str}"


def _mock_query_keyword_extract(prompt_args: Dict) -> str:
    """Mock query keyword extract."""
    keywords_str = ["keyword"] * len(prompt_args["max_keywords"])
    return f"KEYWORDS: {keywords_str}"


# TODO: match prompts based on more than just keywords
# This is very flaky, will fail as we introduce more prompts
PROMPT_TO_INPUT_VARS_MAP = {
    "summary": ["text"],
    "insert": ["num_chunks", "context_list", "new_chunk_text"],
    "query": ["num_chunks", "context_list", "query_str"],
    "query_multiple": ["num_chunks", "context_list", "query_str", "branching_factor"],
    "refine": ["existing_answer", "context_list", "query_str"],
    "text_qa": ["context_str", "query_str"],
    "keyword_extract": ["text", "max_keywords"],
    "query_keyword_extract": ["question", "max_keywords"],
}

PROMPT_TO_FNS_MAP = {
}


class MockLLMPredictor(LLMPredictor):
    """Mock LLM Predictor."""

    def __init__(self, max_tokens: int = NUM_OUTPUTS) -> None:
        """Initialize params."""
        # NOTE: don't pass in LLM
        self.max_tokens = max_tokens


    def _predict(self, prompt: Prompt, **prompt_args: Any) -> Tuple[str, str]:
        """Mock predict."""
        return "mock prediction"
