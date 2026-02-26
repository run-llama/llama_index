"""Row preparation for RLMEnv consumption.

Transforms raw ``DatasetRow`` objects into the format expected by the
RLMEnv runtime (system prompt, task prompt, tools, rubric).
"""

from __future__ import annotations

from typing import Any, Dict, List

from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.runtime.prompts import (
    format_task_prompt,
    get_root_system_prompt,
    get_sub_system_prompt,
)
from environments.rubric_discovery.env.runtime.rubric import build_rubric
from environments.rubric_discovery.env.types import (
    DatasetRow,
    LabeledExample,
    RubricDiscoveryConfig,
)


def transform_row(
    row: DatasetRow,
    config: RubricDiscoveryConfig,
    eval_service: EvaluationService,
) -> Dict[str, Any]:
    """Transform a ``DatasetRow`` into an RLMEnv-compatible episode dict.

    The returned dict contains everything needed to run one episode:
    - system prompt
    - task prompt (with training examples)
    - rubric scorer (for held-out evaluation)
    - configuration metadata

    Args:
        row: A single dataset row with train and test examples.
        config: Environment configuration.
        eval_service: Shared evaluation service instance.

    Returns:
        A dictionary ready for RLMEnv episode construction.
    """
    system_prompt = get_root_system_prompt(config.root_prompt_verbosity)
    task_prompt = format_task_prompt(
        row.train_examples, config.root_prompt_verbosity
    )
    sub_system_prompt = get_sub_system_prompt(config.sub_prompt_verbosity)

    rubric_scorer = build_rubric(
        config=config,
        test_examples=row.test_examples,
        eval_service=eval_service,
    )

    return {
        "system_prompt": system_prompt,
        "task_prompt": task_prompt,
        "sub_system_prompt": sub_system_prompt,
        "train_examples": row.train_examples,
        "test_examples": row.test_examples,
        "rubric_scorer": rubric_scorer,
        "category": row.category,
        "metadata": row.metadata,
        "config": {
            "max_turns": config.max_turns,
            "rlm_model": config.rlm_model,
            "code_execution_timeout": config.code_execution_timeout,
            "max_sub_llm_parallelism": config.max_sub_llm_parallelism,
        },
    }


def prepare_episodes(
    rows: List[DatasetRow],
    config: RubricDiscoveryConfig,
    eval_service: EvaluationService,
) -> List[Dict[str, Any]]:
    """Transform all dataset rows into RLMEnv episodes.

    Args:
        rows: List of dataset rows.
        config: Environment configuration.
        eval_service: Shared evaluation service instance.

    Returns:
        A list of episode dicts, one per row.
    """
    return [transform_row(row, config, eval_service) for row in rows]
