"""Base types for ReAct agent."""

from pydantic import BaseModel
from abc import abstractmethod


class BaseReasoningStep(BaseModel):
    """Reasoning step."""

    @abstractmethod
    def get_content(self) -> str:
        """Get content."""

    @abstractmethod
    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""


class QuestionReasoningStep(BaseModel):
    question: str

    def get_content(self) -> str:
        """Get content."""
        return self.question

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ActionReasoningStep(BaseModel):
    """Action Reasoning step."""

    action: str
    action_input: str

    def get_content(self) -> str:
        """Get content."""
        return f"Action: {self.action}\nAction Input: {self.action_input}"

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ObservationReasoningStep(BaseModel):
    """Action Reasoning step."""

    observation: str

    def get_content(self) -> str:
        """Get content."""
        return self.observation

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ObservationReasoningStep(BaseModel):
    """Action Reasoning step."""

    action: str
    action_input: str
    observation: str

    def get_content(self) -> str:
        """Get content."""
        return self.observation

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return False


class ResponseReasoningStep(BaseModel):
    """Response reasoning step."""

    response: str

    def get_content(self) -> str:
        """Get content."""
        return self.response

    @property
    def is_done(self) -> bool:
        """Is the reasoning step the last one."""
        return True
