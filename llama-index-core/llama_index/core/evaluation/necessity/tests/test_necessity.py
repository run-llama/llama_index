import asyncio
import pytest

from llama_index.core.evaluation.necessity.oracle import ClaimSupportOracle
from llama_index.core.evaluation.necessity.necessity import (
    ClaimNecessityAnalyzer,
    NecessityResult,
)


# ---------------------------------------------------------------------
# Deterministic test doubles
# ---------------------------------------------------------------------

class DummyLLM:
    """
    Deterministic stub LLM for necessity testing.

    Semantics:
    - Claim "The Eiffel Tower is in Paris." is supported
      IFF the specific supporting context
      "The Eiffel Tower is located in Paris." is present.
    - Claim involving "Berlin" is always unsupported.
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

        # Explicitly unsupported claim
        if "Berlin" in prompt:
            return Response("NO")

        # Context-sensitive support
        if "Paris." in prompt:
                return Response("YES")

        return Response("NO")


# ---------------------------------------------------------------------
# Core necessity semantics
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_necessary_context():
    """
    A context is necessary if removing it makes the claim unsupported.
    """
    oracle = ClaimSupportOracle(llm=DummyLLM())
    analyzer = ClaimNecessityAnalyzer(oracle=oracle)

    contexts = [
        "The Eiffel Tower is located in Paris.",   # necessary
        "The Eiffel Tower is a famous landmark.", # irrelevant
    ]

    result: NecessityResult = await analyzer.analyze(
        claim="The Eiffel Tower is in Paris.",
        contexts=contexts,
    )

    assert result.initially_supported is True
    assert result.necessary_context_indices == [0]


@pytest.mark.asyncio
async def test_claim_not_supported_initially():
    """
    If the claim is not supported with full context,
    necessity analysis must short-circuit.
    """
    oracle = ClaimSupportOracle(llm=DummyLLM())
    analyzer = ClaimNecessityAnalyzer(oracle=oracle)

    result = await analyzer.analyze(
        claim="The Eiffel Tower is in Berlin.",
        contexts=["The Eiffel Tower is in Paris."],
    )

    assert result.initially_supported is True
    assert result.necessary_context_indices == []


# ---------------------------------------------------------------------
# Oracle call complexity guarantees
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_necessity_minimal_oracle_calls():
    """
    Necessity analysis must:
    - call oracle once for baseline
    - call oracle once per context removal
    - never repeat calls
    """

    calls = {"count": 0}

    class CountingLLM:
        async def acomplete(
            self,
            prompt: str,
            temperature: float = 0.0,
            **kwargs,
        ):
            calls["count"] += 1

            class Response:
                def __init__(self, text: str):
                    self.text = text

            return Response("YES")

    oracle = ClaimSupportOracle(llm=CountingLLM(), enable_cache=True)
    analyzer = ClaimNecessityAnalyzer(oracle=oracle)

    contexts = ["C1", "C2", "C3"]

    await analyzer.analyze(
        claim="Some claim",
        contexts=contexts,
    )

    # 1 baseline + N removals
    assert calls["count"] == 1 + len(contexts)


# ---------------------------------------------------------------------
# Scale & safeguard behavior
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_contexts_limit_enforced():
    """
    Analyzer must fail fast if context count exceeds max_contexts.
    """
    oracle = ClaimSupportOracle(llm=DummyLLM())
    analyzer = ClaimNecessityAnalyzer(
        oracle=oracle,
        max_contexts=2,
    )

    with pytest.raises(ValueError):
        await analyzer.analyze(
            claim="Some claim",
            contexts=["C1", "C2", "C3"],
        )


@pytest.mark.asyncio
async def test_chunking_behavior_executes_all_removals():
    """
    Chunked execution must still evaluate every context exactly once.
    """

    calls = {"count": 0}

    class CountingLLM:
        async def acomplete(
            self,
            prompt: str,
            temperature: float = 0.0,
            **kwargs,
        ):
            calls["count"] += 1

            class Response:
                def __init__(self, text: str):
                    self.text = text

            return Response("YES")

    oracle = ClaimSupportOracle(llm=CountingLLM())
    analyzer = ClaimNecessityAnalyzer(
        oracle=oracle,
        chunk_size=1,   # force maximum chunking
        parallelism=1,  # deterministic execution
    )

    contexts = ["C1", "C2", "C3", "C4"]

    await analyzer.analyze(
        claim="Some claim",
        contexts=contexts,
    )

    # 1 baseline + N removals
    assert calls["count"] == 1 + len(contexts)


# ---------------------------------------------------------------------
# Observability & telemetry
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_observer_emits_expected_events():
    """
    Observer hook must emit structured events during analysis.
    """

    events = []

    def observer(event):
        events.append(event["event"])

    oracle = ClaimSupportOracle(llm=DummyLLM())
    analyzer = ClaimNecessityAnalyzer(
        oracle=oracle,
        observer=observer,
    )

    await analyzer.analyze(
        claim="The Eiffel Tower is in Paris.",
        contexts=["The Eiffel Tower is located in Paris."],
    )

    assert "baseline_checked" in events
    assert "analysis_completed" in events


@pytest.mark.asyncio
async def test_parallelism_cap_enforced():
    """
    Analyzer must respect the configured parallelism cap.
    """

    active = {"current": 0, "max": 0}

    class TrackingLLM:
        async def acomplete(
            self,
            prompt: str,
            temperature: float = 0.0,
            **kwargs,
        ):
            active["current"] += 1
            active["max"] = max(active["max"], active["current"])

            await asyncio.sleep(0.01)

            active["current"] -= 1

            class Response:
                def __init__(self, text: str):
                    self.text = text

            return Response("YES")

    oracle = ClaimSupportOracle(llm=TrackingLLM())
    analyzer = ClaimNecessityAnalyzer(
        oracle=oracle,
        parallelism=2,
        chunk_size=10,
    )

    contexts = [f"C{i}" for i in range(6)]

    await analyzer.analyze(
        claim="Some claim",
        contexts=contexts,
    )

    assert active["max"] <= 2


@pytest.mark.asyncio
async def test_failure_mode_propagates_from_oracle():
    """
    If the oracle reports failure,
    necessity analysis must fail closed.
    """

    class FailingLLM:
        async def acomplete(
            self,
            prompt: str,
            temperature: float = 0.0,
            **kwargs,
        ):
            raise RuntimeError("simulated failure")

    oracle = ClaimSupportOracle(
        llm=FailingLLM(),
        max_attempts=1,
    )

    analyzer = ClaimNecessityAnalyzer(oracle=oracle)

    result = await analyzer.analyze(
        claim="Some claim",
        contexts=["Some context"],
    )

    assert result.initially_supported is False
    assert result.necessary_context_indices == []
