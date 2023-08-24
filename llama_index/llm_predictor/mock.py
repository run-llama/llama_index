"""Mock LLM Predictor."""
from typing import Any, Dict

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field

from llama_index.constants import DEFAULT_NUM_OUTPUTS
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.llms.base import LLM, LLMMetadata
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.token_counter.utils import (
    mock_extract_keywords_response,
    mock_extract_kg_triplets_response,
)
from llama_index.types import TokenAsyncGen, TokenGen
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
    return mock_extract_kg_triplets_response(prompt_args["text"], max_triplets=max_triplets)


class MockLLMPredictor(BaseLLMPredictor):
    """Mock LLM Predictor."""

    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS, description="Number of tokens to mock generate."
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @property
    def llm(self) -> LLM:
        raise NotImplementedError("MockLLMPredictor does not have an LLM model.")

    def predict(self, prompt: Prompt, **prompt_args: Any) -> str:
        """Mock predict."""

        prompt_str = prompt.prompt_type
        if prompt_str == PromptType.SUMMARY:
            output = _mock_summary_predict(self.max_tokens, prompt_args)
        elif prompt_str == PromptType.TREE_INSERT:
            output = _mock_insert_predict()
        elif prompt_str == PromptType.TREE_SELECT:
            output = _mock_query_select()
        elif prompt_str == PromptType.TREE_SELECT_MULTIPLE:
            output = _mock_query_select_multiple(prompt_args["num_chunks"])
        elif prompt_str == PromptType.REFINE:
            output = _mock_refine(self.max_tokens, prompt, prompt_args)
        elif prompt_str == PromptType.QUESTION_ANSWER:
            output = _mock_answer(self.max_tokens, prompt_args)
        elif prompt_str == PromptType.KEYWORD_EXTRACT:
            output = _mock_keyword_extract(prompt_args)
        elif prompt_str == PromptType.QUERY_KEYWORD_EXTRACT:
            output = _mock_query_keyword_extract(prompt_args)
        elif prompt_str == PromptType.KNOWLEDGE_TRIPLET_EXTRACT:
            output = _mock_knowledge_graph_triplet_extract(
                prompt_args, prompt.partial_dict.get("max_knowledge_triplets", 2)
            )
        elif prompt_str == PromptType.CUSTOM:
            # we don't know specific prompt type, return generic response
            output = ""
        else:
            raise ValueError("Invalid prompt type.")

        return output

    def stream(self, prompt: Prompt, **prompt_args: Any) -> TokenGen:
        raise NotImplementedError

    async def apredict(self, prompt: Prompt, **prompt_args: Any) -> str:
        return self.predict(prompt, **prompt_args)

    async def astream(self, prompt: Prompt, **prompt_args: Any) -> TokenAsyncGen:
        raise NotImplementedError
