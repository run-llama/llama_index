"""Root-tool wrappers: test_rubric, score_examples.

These functions are designed to be registered as tools in the RLMEnv
tool loop, giving the model the ability to test and refine its rubric.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from environments.rubric_discovery.env.candidate.extractor import (
    extract_rubric_fn_source,
)
from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.types import (
    EvaluationResult,
    LabeledExample,
    RubricDiscoveryConfig,
)


class ToolContext:
    """Shared state for tool invocations within a single episode.

    Tracks the latest rubric source and the number of tool calls
    made so far (used by the iteration reward).
    """

    def __init__(
        self,
        config: RubricDiscoveryConfig,
        train_examples: List[LabeledExample],
        eval_service: EvaluationService,
    ) -> None:
        self.config = config
        self.train_examples = train_examples
        self.eval_service = eval_service
        self.latest_source: Optional[str] = None
        self.tool_call_count: int = 0


def evaluate_rubric(ctx: ToolContext, code: str) -> str:
    """Evaluate a candidate rubric_fn against the training examples.

    Args:
        ctx: The current tool context.
        code: Python source code containing a ``rubric_fn`` definition.

    Returns:
        JSON string with per-example results and summary statistics.
    """
    ctx.tool_call_count += 1

    source = extract_rubric_fn_source(code)
    if source is None:
        return json.dumps({
            "success": False,
            "error": "Could not find a `rubric_fn` definition in the provided code.",
        })

    ctx.latest_source = source

    # Validate first
    validation = ctx.eval_service.validate(source)
    if not validation["valid"]:
        return json.dumps({
            "success": False,
            "error": f"Validation failed: {validation.get('error', 'unknown')}",
            "validation": {k: v for k, v in validation.items() if k != "error"},
        })

    # Run evaluation
    result = ctx.eval_service.evaluate(
        source,
        ctx.train_examples,
        timeout_s=ctx.config.code_execution_timeout,
    )

    if not result.valid:
        return json.dumps({
            "success": False,
            "error": result.error or "Evaluation failed.",
        })

    # Build per-example breakdown
    details = []
    for i, (ex, pred) in enumerate(
        zip(ctx.train_examples, result.predictions)
    ):
        details.append({
            "index": i,
            "input_text": ex.input_text[:100],
            "response": ex.response[:100],
            "label": ex.score,
            "prediction": round(pred, 4),
            "error": round(abs(pred - ex.score), 4),
            "within_tolerance": abs(pred - ex.score) < ctx.config.eval_tolerance,
        })

    accuracy = ctx.eval_service.score_within_tolerance(result)
    mae = sum(abs(p - l) for p, l in zip(result.predictions, result.labels)) / max(
        len(result.labels), 1
    )

    return json.dumps({
        "success": True,
        "num_examples": len(ctx.train_examples),
        "accuracy": round(accuracy, 4),
        "mae": round(mae, 4),
        "tolerance": ctx.config.eval_tolerance,
        "details": details,
    })


def score_examples(
    ctx: ToolContext,
    code: str,
    indices: Optional[List[int]] = None,
) -> str:
    """Quick-score a subset of training examples with a candidate rubric.

    Args:
        ctx: The current tool context.
        code: Python source code containing a ``rubric_fn`` definition.
        indices: Optional list of example indices to score. If None,
                 scores all training examples.

    Returns:
        JSON string with predictions for the requested examples.
    """
    ctx.tool_call_count += 1

    source = extract_rubric_fn_source(code)
    if source is None:
        return json.dumps({
            "success": False,
            "error": "Could not find a `rubric_fn` definition in the provided code.",
        })

    ctx.latest_source = source

    # Select subset
    if indices is not None:
        examples = []
        for idx in indices:
            if 0 <= idx < len(ctx.train_examples):
                examples.append(ctx.train_examples[idx])
    else:
        examples = ctx.train_examples

    if not examples:
        return json.dumps({
            "success": False,
            "error": "No valid examples selected.",
        })

    result = ctx.eval_service.evaluate(
        source,
        examples,
        timeout_s=ctx.config.code_execution_timeout,
    )

    if not result.valid:
        return json.dumps({
            "success": False,
            "error": result.error or "Evaluation failed.",
        })

    scores = []
    for ex, pred in zip(examples, result.predictions):
        scores.append({
            "input_text": ex.input_text[:100],
            "label": ex.score,
            "prediction": round(pred, 4),
        })

    return json.dumps({
        "success": True,
        "scores": scores,
    })
