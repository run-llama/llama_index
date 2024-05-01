"""Self Reflection Agent Worker."""

import logging
from typing import Optional

from llama_index.core.agent.types import (
    BaseAgentWorker,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr
from llama_index.core.callbacks import (
    CallbackManager,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

REFLECTION_PROMPT_TEMPLATE = """
You are responsible for evaluating whether an agent is taking the right steps towards a solution.

You are given the current conversation history, which contains the user task, assistant responses + tool calls, \
as well as any feedback that you have already given.

Evaluate the following criteria:
- Whether the tool call arguments make sense
    - Specifically, check whether page numbers are specified when they shouldn't have. They should ONLY be specified
    if in the user query. Do NOT return done if this is the case.
- Whether the tool output completes the task.
- Whether the final message is an ASSISTANT message (not a tool message). Only if the final message
    is an assistant message does it mean the agent is done thinking.

Given the current chat history, please output a reflection response in the following format evaluating
the quality of the agent trajectory:

"""


CORRECTION_PROMPT_TEMPLATE = """
Here is a reflection on the current trajectory.

{reflection_output}

If is_done is not True, there should be feedback on what is going wrong.
Given the feedback, please try again.
"""


class SelfReflectionAgentWorker(BaseModel, BaseAgentWorker):
    """Self Reflection Agent Worker.

    This agent performs a reflection without any tools on a given response
    and subsequently performs correction.
    """

    callback_manager: CallbackManager = Field(default=CallbackManager([]))
    _max_iterations: int = Field(default=5)
    _verbose: bool = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ) -> None:
        ...
