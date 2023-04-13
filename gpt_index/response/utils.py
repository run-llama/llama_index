"""Utilities for response."""

from typing import Generator

from rouge_score import rouge_scorer


def get_response_text(response_gen: Generator) -> str:
    """Get response text."""
    response_text = ""
    for response in response_gen:
        response_text += response
    return response_text


def get_response_context_rouge_score(
    response: str, context: str, metric: str = "rougeL"
) -> float:
    """Caluclate the ROUGE score to measure overlap between response and context."""
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    scores = scorer.score(response, context)
    return scores[metric].recall
