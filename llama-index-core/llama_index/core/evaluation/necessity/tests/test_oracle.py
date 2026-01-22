import pytest

from llama_index.core.evaluation.necessity.oracle import (
    ClaimSupportOracle,
    ClaimSupportResult,
)


# ---------------------------------------------------------------------
# Deterministic test double (claim-blind, structure-aware)
# ---------------------------------------------------------------------

class DeterministicLLM:
    """
    Deterministic stub LLM for claim support testing.

    Semantics:
    - Returns YES iff the *context block* contains evidence ("Paris")
    - Ignores claim wording entirely (claim-blind by design)
    """

    async def acomplete(
        self,
        prompt: str,
        temperature: float = 0.0,
        **kwargs,
    ):
        class Response:
            def __init__(self, text: str):
                self.text = text

        # Robust: depends on structural context, not brittle strings
        if "CONTEXT" in prompt and "Paris" in prompt:
            return Response("YES")

        return Response("NO")


# ---------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claim_supported_when_context_contains_evidence():
    """
    Oracle returns supported=True when the context
    contains evidence supporting some claim.
    """
    oracle = ClaimSupportOracle(llm=DeterministicLLM())

    result = await oracle.check(
        claim="The Eiffel Tower is in Paris.",
        contexts=[
            "The Eiffel Tower is located in Paris, France."
        ],
    )

    assert isinstance(result, ClaimSupportResult)
    assert result.supported is True
    assert result.raw_response.upper().startswith("YES")


@pytest.mark.asyncio
async def test_claim_not_supported_when_context_lacks_evidence():
    """
    Oracle returns supported=False when the context
    does not contain supporting evidence.
    """
    oracle = ClaimSupportOracle(llm=DeterministicLLM())

    result = await oracle.check(
        claim="Any arbitrary claim",
        contexts=[
            "The Eiffel Tower was completed in 1889."
        ],
    )

    assert isinstance(result, ClaimSupportResult)
    assert result.supported is False


@pytest.mark.asyncio
async def test_empty_context_fails_closed():
    """
    Oracle must fail closed (supported=False)
    when no context is provided.
    """
    oracle = ClaimSupportOracle(llm=DeterministicLLM())

    result = await oracle.check(
        claim="Some factual claim",
        contexts=[],
    )

    assert result.supported is False


# ---------------------------------------------------------------------
# Caching behavior tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oracle_caching():
    """
    Oracle must cache deterministic results
    to avoid repeated LLM calls.
    """

    call_counter = {"count": 0}

    class CountingLLM:
        async def acomplete(
            self,
            prompt: str,
            temperature: float = 0.0,
            **kwargs,
        ):
            call_counter["count"] += 1

            class Response:
                def __init__(self, text: str):
                    self.text = text

            return Response("YES")

    oracle = ClaimSupportOracle(
        llm=CountingLLM(),
        enable_cache=True,
    )

    claim = "Test claim"
    contexts = ["Some supporting context"]

    # First call → LLM invocation expected
    first = await oracle.check(
        claim=claim,
        contexts=contexts,
    )

    # Second call → must be served from cache
    second = await oracle.check(
        claim=claim,
        contexts=contexts,
    )

    assert first.supported is True
    assert second.supported is True

    # Critical invariant: only one LLM call
    assert call_counter["count"] == 1
