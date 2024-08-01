"""Mock utils for query transform."""

from llama_index.core.indices.query.query_transform.prompts import (
    DecomposeQueryTransformPrompt,
)
from llama_index.core.prompts.prompt_type import PromptType

MOCK_DECOMPOSE_TMPL = "{context_str}\n{query_str}"
MOCK_DECOMPOSE_PROMPT = DecomposeQueryTransformPrompt(
    MOCK_DECOMPOSE_TMPL, prompt_type=PromptType.DECOMPOSE
)
