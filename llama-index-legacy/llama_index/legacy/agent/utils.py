"""Agent utils."""

from llama_index.legacy.agent.types import TaskStep
from llama_index.legacy.core.llms.types import MessageRole
from llama_index.legacy.llms.base import ChatMessage
from llama_index.legacy.memory import BaseMemory


def add_user_step_to_memory(
    step: TaskStep, memory: BaseMemory, verbose: bool = False
) -> None:
    """Add user step to memory."""
    user_message = ChatMessage(content=step.input, role=MessageRole.USER)
    memory.put(user_message)
    if verbose:
        print(f"Added user message to memory: {step.input}")
