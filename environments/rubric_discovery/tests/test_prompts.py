"""Tests for env/runtime/prompts.py."""

from environments.rubric_discovery.env.runtime.prompts import (
    format_task_prompt,
    get_root_system_prompt,
    get_sub_system_prompt,
)
from environments.rubric_discovery.env.types import LabeledExample


class TestGetRootSystemPrompt:
    def test_light(self) -> None:
        prompt = get_root_system_prompt("light")
        assert "scoring-rule discovery agent" in prompt
        assert len(prompt) < 300

    def test_medium(self) -> None:
        prompt = get_root_system_prompt("medium")
        assert "rubric_fn" in prompt

    def test_heavy(self) -> None:
        prompt = get_root_system_prompt("heavy")
        assert "rubric_fn" in prompt
        assert "Strategy" in prompt
        assert "Tips" in prompt

    def test_default(self) -> None:
        prompt = get_root_system_prompt("unknown")
        assert prompt == get_root_system_prompt("heavy")


class TestGetSubSystemPrompt:
    def test_levels(self) -> None:
        for level in ("light", "medium", "heavy"):
            prompt = get_sub_system_prompt(level)
            assert len(prompt) > 0


class TestFormatTaskPrompt:
    def test_includes_examples(self) -> None:
        examples = [
            LabeledExample("q1", "a1", 0.5),
            LabeledExample("q2", "a2", 1.0),
        ]
        prompt = format_task_prompt(examples, "heavy")
        assert "q1" in prompt
        assert "a1" in prompt
        assert "0.5" in prompt
        assert "Example 1" in prompt
        assert "Example 2" in prompt

    def test_light(self) -> None:
        examples = [LabeledExample("q", "a", 0.5)]
        prompt = format_task_prompt(examples, "light")
        assert "q" in prompt
