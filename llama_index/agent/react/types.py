"""Base types for ReAct agent."""

from pydantic import BaseModel
from abc import abstractmethod
from typing import Dict


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""


class QuestionReasoningStep(BaseReasoningStep):
    question: str

    def get_content(self) -> str:
        """Get content."""
        return f"Question: {self.question}\n"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ActionReasoningStep(BaseReasoningStep):
    """Action Reasoning step."""

    thought: str
    action: str
    action_input: Dict

    def get_content(self) -> str:
        """Get content."""
        return (
            f"Thought: {self.thought}\nAction: {self.action}\n"
            f"Action Input: {self.action_input}"
        )

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ObservationReasoningStep(BaseReasoningStep):
    """Action Reasoning step."""

    observation: str

    def get_content(self) -> str:
        """Get content."""
        return f"Observation: {self.observation}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ResponseReasoningStep(BaseReasoningStep):
    """Response reasoning step."""

    thought: str
    response: str

    def get_content(self) -> str:
        """Get content."""
        return f"Response: {self.response}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return True
