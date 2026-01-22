import pytest

from llama_index.core.evaluation.necessity.dataset import (
    DatasetExample,
    NecessityDatasetRunner,
)
from llama_index.core.evaluation.necessity.evaluator import (
    EvidenceNecessityEvaluator,
)
from llama_index.core.evaluation.necessity.oracle import (
    ClaimSupportOracle,
)


class DeterministicLLM:
    async def acomplete(self, prompt: str, temperature: float = 0.0, **kwargs):
        class R:
            def __init__(self, text: str):
                self.text = text

        if "Paris" in prompt:
            return R("YES")
        return R("NO")


@pytest.mark.asyncio
async def test_dataset_runner():
    oracle = ClaimSupportOracle(llm=DeterministicLLM())
    evaluator = EvidenceNecessityEvaluator(oracle=oracle)
    runner = NecessityDatasetRunner(evaluator=evaluator)

    dataset = [
        DatasetExample(
            query="Where is the Eiffel Tower?",
            response="The Eiffel Tower is in Paris.",
            contexts=[
                "The Eiffel Tower is located in Paris.",
                "The Eiffel Tower is a tourist attraction.",
            ],
        )
    ]

    result = await runner.run(dataset)

    assert result.examples == 1
    assert result.unsupported_claims == 0
    assert result.fragile_claims == 1
    assert result.robust_claims == 0
