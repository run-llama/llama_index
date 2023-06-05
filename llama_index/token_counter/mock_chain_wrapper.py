"""Mock chain wrapper."""

from typing import Any, Dict, Optional

from langchain.llms.base import BaseLLM

from llama_index.constants import DEFAULT_NUM_OUTPUTS
from llama_index.langchain_helpers.chain_wrapper import LLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.token_counter.utils import (
    mock_extract_keywords_response,
    mock_extract_kg_triplets_response,
)
from llama_index.utils import globals_helper

# TODO: consolidate with unit tests in tests/mock_utils/mock_predict.py


def _mock_summary_predict(max_tokens: int, prompt_args: Dict) -> str:
    """Mock summary predict."""
    # tokens in response shouldn't be larger than tokens in `context_str`
    num_text_tokens = len(globals_helper.tokenizer(prompt_args["context_str"]))
    token_limit = min(num_text_tokens, max_tokens)
    return " ".join(["summary"] * token_limit)


def _mock_insert_predict() -> str:
    """Mock insert predict."""
    return "ANSWER: 1"


def _mock_query_select() -> str:
    """Mock query select."""
    return "ANSWER: 1"


def _mock_query_select_multiple(num_chunks: int) -> str:
    """Mock query select."""
    nums_str = ", ".join([str(i) for i in range(num_chunks)])
    return f"ANSWER: {nums_str}"


def _mock_answer(max_tokens: int, prompt_args: Dict) -> str:
    """Mock answer."""
    # tokens in response shouldn't be larger than tokens in `text`
    num_ctx_tokens = len(globals_helper.tokenizer(prompt_args["context_str"]))
    token_limit = min(num_ctx_tokens, max_tokens)
    return " ".join(["answer"] * token_limit)


def _mock_refine(max_tokens: int, prompt: Prompt, prompt_args: Dict) -> str:
    """Mock refine."""
    # tokens in response shouldn't be larger than tokens in
    # `existing_answer` + `context_msg`
    # NOTE: if existing_answer is not in prompt_args, we need to get it from the prompt
    if "existing_answer" not in prompt_args:
        existing_answer = prompt.partial_dict["existing_answer"]
    else:
        existing_answer = prompt_args["existing_answer"]
    num_ctx_tokens = len(globals_helper.tokenizer(prompt_args["context_msg"]))
    num_exist_tokens = len(globals_helper.tokenizer(existing_answer))
    token_limit = min(num_ctx_tokens + num_exist_tokens, max_tokens)
    return " ".join(["answer"] * token_limit)


def _mock_keyword_extract(prompt_args: Dict) -> str:
    """Mock keyword extract."""
    return mock_extract_keywords_response(prompt_args["text"])


def _mock_query_keyword_extract(prompt_args: Dict) -> str:
    """Mock query keyword extract."""
    return mock_extract_keywords_response(prompt_args["question"])


def _mock_knowledge_graph_triplet_extract(prompt_args: Dict, max_triplets: int) -> str:
    """Mock knowledge graph triplet extract."""
    return mock_extract_kg_triplets_response(
        prompt_args["text"], max_triplets=max_triplets
    )


class MockLLMPredictor(LLMPredictor):
    """Mock LLM Predictor."""

    def __init__(
        self, max_tokens: int = DEFAULT_NUM_OUTPUTS, llm: Optional[BaseLLM] = None
    ) -> None:
        """Initialize params."""
        super().__init__(llm)
        # NOTE: don't call super, we don't want to instantiate LLM
        self.max_tokens = max_tokens
        self._total_tokens_used = 0
        self.flag = True
        self._last_token_usage = None

    def _predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Mock predict."""
        prompt_str = prompt.prompt_type
        if prompt_str == PromptType.SUMMARY:
            return _mock_summary_predict(self.max_tokens, prompt_args)
        elif prompt_str == PromptType.TREE_INSERT:
            return _mock_insert_predict()
        elif prompt_str == PromptType.TREE_SELECT:
            return _mock_query_select()
        elif prompt_str == PromptType.TREE_SELECT_MULTIPLE:
            return _mock_query_select_multiple(prompt_args["num_chunks"])
        elif prompt_str == PromptType.REFINE:
            return _mock_refine(self.max_tokens, prompt, prompt_args)
        elif prompt_str == PromptType.QUESTION_ANSWER:
            return _mock_answer(self.max_tokens, prompt_args)
        elif prompt_str == PromptType.KEYWORD_EXTRACT:
            return _mock_keyword_extract(prompt_args)
        elif prompt_str == PromptType.QUERY_KEYWORD_EXTRACT:
            return _mock_query_keyword_extract(prompt_args)
        elif prompt_str == PromptType.KNOWLEDGE_TRIPLET_EXTRACT:
            return _mock_knowledge_graph_triplet_extract(
                prompt_args, prompt.partial_dict.get("max_knowledge_triplets", 2)
            )
        elif prompt_str == PromptType.CUSTOM:
            # we don't know specific prompt type, return generic response
            return ""
        else:
            raise ValueError("Invalid prompt type.")
