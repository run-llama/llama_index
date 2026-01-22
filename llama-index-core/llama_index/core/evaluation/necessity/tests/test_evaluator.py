import pytest
import json
from llama_index.core.evaluation.necessity.evaluator import (
    EvidenceNecessityEvaluator,
)
from llama_index.core.evaluation.necessity.oracle import ClaimSupportOracle


class DummyLLM:
    async def acomplete(self, prompt: str, temperature: float = 0.0, **kwargs):
        class R:
            def __init__(self, text: str):
                self.text = text

        if "Paris" in prompt and "Eiffel Tower" in prompt:
            return R("YES")
        return R("NO")


@pytest.mark.asyncio
async def test_evidence_necessity_evaluator():
    oracle = ClaimSupportOracle(llm=DummyLLM())
    evaluator = EvidenceNecessityEvaluator(oracle=oracle)

    result = await evaluator.aevaluate(
        query="Where is the Eiffel Tower?",
        response="The Eiffel Tower is in Paris.",
        contexts=[
            "The Eiffel Tower is located in Paris.",
            "The Eiffel Tower is a famous landmark.",
        ],
    )

    assert result.invalid_result is False
    feedback = json.loads(result.feedback)
    assert "claims" in feedback
    assert (
        feedback["claims"]
        ["The Eiffel Tower is in Paris."]["supported"]
        is True
    )
