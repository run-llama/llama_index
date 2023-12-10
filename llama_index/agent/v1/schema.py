from pydantic import BaseModel, Field
from llama_index.memory.types import BaseMemory
from typing import List, Any, Dict, Deque
from collections import deque
from abc import ABC, abstractmethod


class TaskStep(BaseModel):
    """"Agent task step.

    Represents a single input step within the execution run ("Task") of an agent
    given a user input.

    The output is returned as a `TaskStepOutput`.
    
    """
    task_id: str = Field(..., type=str, description="Task ID")
    step_id: str = Field(..., type=str, description="Step ID")
    input: str = Field(..., type=str, description="User input")
    memory: BaseMemory = Field(..., type=BaseMemory, description="Memory")


class TaskStepOutput(BaseModel):
    """Agent task step output."""
    output: Any = Field(..., type=Any, description="Task step output")
    task_step: TaskStep = Field(..., type=TaskStep, description="Task step input")
    is_last: bool = Field(default=False, description="Is this the last step?")

    def __str__(self) -> str:
        return str(self.output)


class TaskState(BaseModel):
    """Get task state."""
    step_queue: Deque[TaskStep] = Field(default_factory=deque, description="Task step queue.")
    completed_steps: List[TaskStepOutput] = Field(..., description="Completed step outputs.")
    

class Task(BaseModel):
    """Agent Task.

    Represents a "run" of an agent given a user input.
    
    """
    task_id: str = Field(..., type=str, description="Task ID")
    input: str = Field(..., type=str, description="User input")
    task_state: TaskState = Field(..., description="Current task state")



class AgentState(BaseModel):
    """Agent state."""
    
    task_state_dict: Dict[str, TaskState] = Field(default_factory=dict, description="Task state dictionary.")
    
    def get_task_state(self, task_id: str) -> TaskState:
        """Get task state."""
        return self.task_state_dict[task_id]

    def get_completed_steps(self, task_id: str) -> List[TaskStepOutput]:
        """Get completed steps."""
        return self.get_task_state(task_id).completed_steps

    def get_step_queue(self, task_id: str) -> List[TaskStep]:
        """Get step queue."""
        return self.get_task_state(task_id).step_queue


class BaseAgentStepEngine(ABC):
    """Base agent step engine."""
    
    @abstractmethod
    def _run_step(
        self, 
        step: TaskStep, 
        step_queue: Deque[TaskStep],
        **kwargs: Any
    ) -> TaskStepOutput:
        """Run step."""

    def run_step(
        self, 
        step: TaskStep, 
        step_queue: Deque[TaskStep],
        **kwargs: Any
    ) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, step_queue, **kwargs)
    
