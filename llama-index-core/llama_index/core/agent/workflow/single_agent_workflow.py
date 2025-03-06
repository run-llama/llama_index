from abc import ABC
from typing import Any, List, Optional, Union, TypeVar

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.workflow import (
    Context,
)
from llama_index.core.workflow.checkpointer import CheckpointCallback
from llama_index.core.workflow.handler import WorkflowHandler

T = TypeVar("T", bound="BaseWorkflowAgent")  # type: ignore[name-defined]


class SingleAgentRunnerMixin(ABC):
    """Mixin class for executing a single agent within a workflow system.
    This class provides the necessary interface for running a single agent.
    """

    def run(
        self: T,
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        ctx: Optional[Context] = None,
        stepwise: bool = False,
        checkpoint_callback: Optional[CheckpointCallback] = None,
        **workflow_kwargs: Any,
    ) -> WorkflowHandler:
        """Run the agent."""
        from llama_index.core.agent.workflow import AgentWorkflow

        workflow = AgentWorkflow(agents=[self], **workflow_kwargs)
        return workflow.run(
            user_msg=user_msg,
            chat_history=chat_history,
            memory=memory,
            ctx=ctx,
            stepwise=stepwise,
            checkpoint_callback=checkpoint_callback,
        )
