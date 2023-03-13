"""Mock utils for query transform."""

from gpt_index.indices.query.query_transform.prompts import (
    DecomposeQueryTransformPrompt,
)

MOCK_DECOMPOSE_TMPL = "{context_str}\n{query_str}"
MOCK_DECOMPOSE_PROMPT = DecomposeQueryTransformPrompt(MOCK_DECOMPOSE_TMPL)
