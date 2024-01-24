# ReAct agent formatter

import logging
from abc import abstractmethod
from typing import List, Optional, Sequence

from llama_index.core.agent.react.types import BaseReasoningStep, ObservationReasoningStep
from llama_index.bridge.pydantic import BaseModel
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.tools import BaseTool
from llama_index.chat_engine.react.prompts import REACT_CHAT_ENGINE_SYSTEM_HEADER

logger = logging.getLogger(__name__)


class ReActChatEngineFormatter:
    """ReAct chat engine formatter.

    Simpler version of `ReActChatFormatter`. No tool use or context.
    
    """

    system_header: str = REACT_CHAT_ENGINE_SYSTEM_HEADER  # default

    def format(
        self,
        data_desc: str,
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> List[ChatMessage]:
        current_reasoning = current_reasoning or []

        fmt_sys_header = self.system_header.format(data_desc=data_desc)

        # format reasoning history as alternating user and assistant messages
        # where the assistant messages are thoughts and actions and the user
        # messages are observations
        reasoning_history = []
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=MessageRole.USER,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=fmt_sys_header),
            *chat_history,
            *reasoning_history,
        ]
