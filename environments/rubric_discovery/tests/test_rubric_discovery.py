"""Tests for the main rubric_discovery.py module."""

import json

import pytest

from environments.rubric_discovery.rubric_discovery import (
    RubricDiscoveryEnvironment,
    load_environment,
)
from environments.rubric_discovery.env.types import RubricDiscoveryConfig


class TestLoadEnvironment:
    def test_default_config(self) -> None:
        env = load_environment()
        assert isinstance(env, RubricDiscoveryEnvironment)
        assert env.num_episodes() > 0

    def test_custom_config_dict(self) -> None:
        env = load_environment({"max_turns": 5, "eval_backend": "subprocess"})
        assert env.config.max_turns == 5

    def test_custom_config_object(self) -> None:
        cfg = RubricDiscoveryConfig(max_turns=3)
        env = load_environment(cfg)
        assert env.config.max_turns == 3

    def test_with_categories(self) -> None:
        env = load_environment({"categories": ["math_correctness"]})
        assert env.num_episodes() >= 1
        for ep in env.episodes:
            assert ep["category"] == "math_correctness"


class TestRubricDiscoveryEnvironment:
    @pytest.fixture
    def env(self) -> RubricDiscoveryEnvironment:
        return load_environment({"eval_backend": "subprocess"})

    def test_get_episode(self, env: RubricDiscoveryEnvironment) -> None:
        ep = env.get_episode(0)
        assert "system_prompt" in ep
        assert "task_prompt" in ep
        assert "rubric_scorer" in ep
        assert "train_examples" in ep
        assert "test_examples" in ep

    def test_create_tool_context(self, env: RubricDiscoveryEnvironment) -> None:
        ctx = env.create_tool_context(0)
        assert ctx.tool_call_count == 0
        assert ctx.latest_source is None
        assert len(ctx.train_examples) > 0

    def test_get_tools(self, env: RubricDiscoveryEnvironment) -> None:
        tools = env.get_tools(0)
        assert "test_rubric" in tools
        assert "score_examples" in tools
        assert "handler" in tools["test_rubric"]

    def test_score_episode_valid(self, env: RubricDiscoveryEnvironment) -> None:
        model_output = """
Here is my rubric:
```python
def rubric_fn(input_text, response):
    return 0.5
```
"""
        result = env.score_episode(0, model_output, num_tool_calls=3)
        assert "reward" in result
        assert "rewards" in result
        assert "metrics" in result
        assert result["metrics"]["valid"]
        assert result["metrics"]["num_tool_calls"] == 3

    def test_score_episode_no_rubric(self, env: RubricDiscoveryEnvironment) -> None:
        result = env.score_episode(0, "I don't know how to write a rubric")
        assert result["reward"] == 0.0
        assert not result["metrics"]["valid"]

    def test_tool_integration(self, env: RubricDiscoveryEnvironment) -> None:
        """Test that tools work end-to-end within an episode."""
        tools = env.get_tools(0)
        code = "def rubric_fn(input_text, response): return 0.5"

        # Call test_rubric
        result = json.loads(tools["test_rubric"]["handler"](code))
        assert result["success"]

        # Call score_examples
        result = json.loads(tools["score_examples"]["handler"](code))
        assert result["success"]
