"""Cached prompt tokens are billed and must be counted.

Anthropic reports `input_tokens` NET of the prompt cache, with the cached tokens in
`cache_read_input_tokens` and `cache_creation_input_tokens`. Reading `input_tokens`
alone therefore drops every token served from cache. With a warm cache that is not a
rounding error: `input_tokens` can be single digits while `cache_read_input_tokens` is
tens of thousands, so `TokenCountingHandler` reported 3 tokens for a call that billed
21,503.

That number feeds the `token_budget` guard, so a budget silently fails to fire.

OpenAI and Gemini already include their cached tokens in `prompt_tokens` /
`prompt_token_count`, so those must NOT be adjusted. The last two tests pin that,
because the obvious fix double-counts them.
"""

from types import SimpleNamespace

from llama_index.core.callbacks.token_counting import get_tokens_from_response


def _response(usage):
    """Minimal stand-in for a ChatResponse carrying a provider usage blob."""
    return SimpleNamespace(raw={"usage": usage}, additional_kwargs={})


class TestAnthropicCacheTokens:
    def test_cache_read_tokens_are_counted(self):
        """A warm cache: 3 uncached tokens, 20000 served from cache. All billed."""
        prompt, completion = get_tokens_from_response(
            _response(
                {
                    "input_tokens": 3,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 20000,
                }
            )
        )
        assert prompt == 20003
        assert completion == 120

    def test_cache_creation_tokens_are_counted(self):
        """Writing the cache is billed too, at a premium."""
        prompt, _ = get_tokens_from_response(
            _response(
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 1500,
                }
            )
        )
        assert prompt == 1510

    def test_both_cache_fields_are_counted(self):
        """The case that motivated this: 3 reported against 21503 billed."""
        prompt, _ = get_tokens_from_response(
            _response(
                {
                    "input_tokens": 3,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 20000,
                    "cache_creation_input_tokens": 1500,
                }
            )
        )
        assert prompt == 21503

    def test_null_cache_fields_are_treated_as_zero(self):
        """Anthropic may send the keys explicitly set to null rather than omitting."""
        prompt, _ = get_tokens_from_response(
            _response(
                {
                    "input_tokens": 42,
                    "output_tokens": 7,
                    "cache_read_input_tokens": None,
                    "cache_creation_input_tokens": None,
                }
            )
        )
        assert prompt == 42

    def test_uncached_anthropic_call_is_unchanged(self):
        """With no cache in play the previous arithmetic was already right."""
        prompt, completion = get_tokens_from_response(
            _response({"input_tokens": 100, "output_tokens": 50})
        )
        assert prompt == 100
        assert completion == 50


class TestProvidersThatAlreadyIncludeCache:
    """These pin the boundary. A fix that just adds any cache count breaks them."""

    def test_openai_prompt_tokens_are_not_adjusted(self):
        """OpenAI's prompt_tokens ALREADY includes prompt_tokens_details.cached_tokens."""
        prompt, completion = get_tokens_from_response(
            _response(
                {
                    "prompt_tokens": 5000,
                    "completion_tokens": 200,
                    "prompt_tokens_details": {"cached_tokens": 4864},
                }
            )
        )
        assert prompt == 5000, "adding the cached tokens here would double-count"
        assert completion == 200

    def test_gemini_prompt_token_count_is_not_adjusted(self):
        """Gemini's prompt_token_count ALREADY includes cached_content_token_count."""
        prompt, completion = get_tokens_from_response(
            _response(
                {
                    "prompt_token_count": 8000,
                    "candidates_token_count": 300,
                    "cached_content_token_count": 7500,
                }
            )
        )
        assert prompt == 8000, "adding the cached tokens here would double-count"
        assert completion == 300
