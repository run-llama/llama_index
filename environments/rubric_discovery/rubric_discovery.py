"""Rubric-discovery environment: learn to synthesize rubric_fn from labeled examples.

This module provides the ``load_environment`` entry point that constructs
the full environment with tools, prompts, dataset, and scoring rubric.

Usage::

    prime eval run rubric-discovery
    prime eval run rubric-discovery \\
        -m gpt-4.1-mini \\
        -a '{"config": {"max_turns": 8, "rlm_model": "gpt-4.1-mini"}}'
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from environments.rubric_discovery.env.dataset.loader import load_dataset
from environments.rubric_discovery.env.dataset.transform import (
    prepare_episodes,
    transform_row,
)
from environments.rubric_discovery.env.evaluation.service import EvaluationService
from environments.rubric_discovery.env.runtime.prompts import (
    format_task_prompt,
    get_root_system_prompt,
    get_sub_system_prompt,
)
from environments.rubric_discovery.env.runtime.rewards import compute_rewards
from environments.rubric_discovery.env.runtime.rubric import RubricScorer, build_rubric
from environments.rubric_discovery.env.runtime.tools import (
    ToolContext,
    score_examples,
    evaluate_rubric,
)
from environments.rubric_discovery.env.types import (
    DatasetRow,
    EvaluationResult,
    ExecutionBackend,
    LabeledExample,
    RubricDiscoveryConfig,
)


class RubricDiscoveryEnvironment:
    """Multi-turn + tools environment for rubric discovery.

    Each episode presents training examples of ``(input, response, score)``
    and asks the model to infer the hidden scoring rule. The model iterates
    with a Python REPL, writes candidate rubrics, and is scored on held-out
    test examples.

    Attributes:
        config: The resolved configuration.
        episodes: Prepared episode data for RLMEnv consumption.
    """

    def __init__(self, config: RubricDiscoveryConfig) -> None:
        self.config = config
        self._eval_service = EvaluationService(config)

        # Load and prepare dataset
        rows = load_dataset(
            path=config.dataset_path,
            categories=config.categories,
            max_examples=config.max_examples,
        )
        self.episodes = prepare_episodes(rows, config, self._eval_service)

    @property
    def eval_service(self) -> EvaluationService:
        """The shared evaluation service."""
        return self._eval_service

    def get_episode(self, index: int) -> Dict[str, Any]:
        """Return the episode dict at *index*."""
        return self.episodes[index]

    def num_episodes(self) -> int:
        """Number of episodes in the dataset."""
        return len(self.episodes)

    def create_tool_context(self, episode_index: int) -> ToolContext:
        """Create a tool context for the given episode.

        This context is shared across tool calls within a single episode
        and tracks state like the latest rubric source and call count.
        """
        episode = self.episodes[episode_index]
        return ToolContext(
            config=self.config,
            train_examples=episode["train_examples"],
            eval_service=self._eval_service,
        )

    def score_episode(
        self,
        episode_index: int,
        model_output: str,
        num_tool_calls: int = 0,
    ) -> Dict[str, Any]:
        """Score the model's output for a given episode.

        Args:
            episode_index: Index of the episode in the dataset.
            model_output: The model's complete output text.
            num_tool_calls: Total tool calls made during the episode.

        Returns:
            Scoring results with reward breakdown and metrics.
        """
        episode = self.episodes[episode_index]
        scorer: RubricScorer = episode["rubric_scorer"]
        return scorer.score(model_output, num_tool_calls)

    def get_tools(self, episode_index: int) -> Dict[str, Any]:
        """Return the tool definitions for an episode.

        The returned dict maps tool names to their metadata, suitable
        for registration in an RLMEnv tool loop.
        """
        ctx = self.create_tool_context(episode_index)
        return {
            "test_rubric": {
                "description": (
                    "Evaluate your candidate rubric_fn against the training "
                    "examples. Pass the complete Python source code as the "
                    "'code' argument."
                ),
                "parameters": {
                    "code": {
                        "type": "string",
                        "description": "Python source code containing a rubric_fn definition.",
                    }
                },
                "handler": lambda code: evaluate_rubric(ctx, code),
            },
            "score_examples": {
                "description": (
                    "Quick-score a subset of examples with your rubric_fn. "
                    "Pass the source code and optionally a list of example "
                    "indices to score."
                ),
                "parameters": {
                    "code": {
                        "type": "string",
                        "description": "Python source code containing a rubric_fn definition.",
                    },
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional list of example indices to score.",
                    },
                },
                "handler": lambda code, indices=None: score_examples(
                    ctx, code, indices
                ),
            },
        }


def load_environment(
    config: Optional[Dict[str, Any]] = None,
) -> RubricDiscoveryEnvironment:
    """Load and return the rubric-discovery environment.

    This is the main entry point called by the Prime evaluation runner.

    Args:
        config: Optional configuration dictionary. Keys correspond to
                ``RubricDiscoveryConfig`` fields. If ``None``, defaults
                are used.

    Returns:
        A fully initialized ``RubricDiscoveryEnvironment``.

    Example::

        # Default configuration
        env = load_environment()

        # Custom configuration
        env = load_environment({
            "max_turns": 8,
            "eval_backend": "subprocess",
            "eval_tolerance": 0.3,
        })
    """
    if config is None:
        cfg = RubricDiscoveryConfig()
    elif isinstance(config, RubricDiscoveryConfig):
        cfg = config
    else:
        cfg = RubricDiscoveryConfig.from_dict(config)

    return RubricDiscoveryEnvironment(cfg)
