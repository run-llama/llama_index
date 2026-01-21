from typing import Any, Optional, Sequence, List, Dict

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.settings import Settings

from llama_index.core.evaluation.counterfactual.perturbations import (
    remove_top_k_contexts,
)
from llama_index.core.evaluation.counterfactual.metrics import (
    lexical_overlap,
    embedding_distance,
    significant_drift,
)


DEFAULT_COUNTERFACTUAL_PROMPT = (
    "Answer the question using ONLY the information below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)


class CounterfactualEvaluator(BaseEvaluator):
    """
    Counterfactual evaluator for RAG systems.

    Evaluates whether a generated response causally depends on retrieved contexts
    by perturbing evidence and measuring answer drift.
    """

    def __init__(
        self,
        llm=None,
        enable_regeneration: bool = True,
        prompt_template: Optional[str] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._enable_regeneration = enable_regeneration
        self._prompt_template = (
            prompt_template or DEFAULT_COUNTERFACTUAL_PROMPT
        )

    async def _generate_answer(
        self,
        query: str,
        contexts: Sequence[str],
    ) -> str:
        """Generate an answer from query + contexts using the LLM."""
        context_block = "\n\n".join(contexts)
        prompt = self._prompt_template.format(
            query=query,
            context=context_block,
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

        if not self._enable_regeneration:
            return EvaluationResult(
                invalid_result=True,
                invalid_reason="Counterfactual regeneration is disabled",
            )

        baseline_answer: str = response
        drift_records: List[Dict[str, float | bool]] = []

        # ---- Counterfactual: remove top-1 context ----
        perturbed_contexts = remove_top_k_contexts(contexts, k=1)

        if perturbed_contexts:
            cf_answer = await self._generate_answer(
                query=query,
                contexts=perturbed_contexts,
            )

            lex_score = lexical_overlap(baseline_answer, cf_answer)
            emb_dist = await embedding_distance(baseline_answer, cf_answer)
            drift_flag = significant_drift(lex_score, emb_dist)

            drift_records.append(
                {
                    "lexical_overlap": lex_score,
                    "embedding_distance": emb_dist,
                    "significant_drift": drift_flag,
                }
            )

        # ---- Aggregate score ----
        score = (
            sum(1.0 for r in drift_records if r["significant_drift"])
            / len(drift_records)
            if drift_records
            else 0.0
        )

        return EvaluationResult(
            query=query,
            response=baseline_answer,
            contexts=contexts,
            score=score,
            feedback="Counterfactual drift evaluation completed",
        )
