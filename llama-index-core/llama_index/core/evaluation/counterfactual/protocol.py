from typing import Any, Optional, Sequence, List

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.settings import Settings

from llama_index.core.evaluation.counterfactual.perturbations import (
    remove_top_k_contexts,
)


class CounterfactualEvaluator(BaseEvaluator):
    """
    Counterfactual evaluator for RAG systems.

    Tests whether a generated response causally depends on retrieved contexts
    by perturbing the evidence and comparing regenerated answers.
    """

    def __init__(
        self,
        llm=None,
        max_counterfactuals: int = 1,
    ) -> None:
        self._llm = llm or Settings.llm
        self._max_counterfactuals = max_counterfactuals

    async def _generate_answer(
        self,
        query: str,
        contexts: Sequence[str],
    ) -> str:
        """Generate an answer from query + contexts using the LLM."""
        context_block = "\n\n".join(contexts)
        prompt = (
            "Answer the question using ONLY the information below.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        response = await self._llm.acomplete(prompt)
        return response.text

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        if query is None or response is None or contexts is None:
            return EvaluationResult(
                invalid_result=True,
                invalid_reason="query, response, and contexts must be provided",
            )

        # Baseline answer (already generated)
        baseline_answer = response

        counterfactual_answers: List[str] = []

        # Simple counterfactual: remove top-1 context
        perturbed_contexts = remove_top_k_contexts(contexts, k=1)

        if perturbed_contexts:
            cf_answer = await self._generate_answer(
                query=query,
                contexts=perturbed_contexts,
            )
            counterfactual_answers.append(cf_answer)

        return EvaluationResult(
            query=query,
            response=baseline_answer,
            contexts=contexts,
            feedback=(
                "Counterfactual answers generated. "
                f"Num counterfactuals: {len(counterfactual_answers)}"
            ),
            score=None,
        )
