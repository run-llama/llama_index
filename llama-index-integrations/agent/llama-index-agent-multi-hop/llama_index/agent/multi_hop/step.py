"""Multi-Hop agent worker."""

import logging
from typing import Any, Coroutine, List, Optional, Sequence

from llama_index.core.agent.types import (
    BaseAgentWorker,
)
from llama_index.core.base.agent.types import Task, TaskStep, TaskStepOutput
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool
from llama_index.core.tools.query_engine import QueryEngineTool

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


class MultiHopAgentWorker(BaseAgentWorker):
    """MultiHop Agent Worker."""

    def __init__(
        self,
        llm: LLM,
        query_engine_tool: QueryEngineTool,
        tools: Optional[Sequence[BaseTool]] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self.llm = llm
        self.query_engine_tool = query_engine_tool
        self.tools = tools
        self.verbose = verbose
        self.callback_manager = callback_manager or CallbackManager([])

    @classmethod
    def from_defaults(
        cls,
        query_engine: BaseQueryEngine,
        llm: Optional[LLM] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> "MultiHopAgentWorker":
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine, **kwargs
        )

        return cls(
            llm=llm,
            query_engine_tool=query_engine_tool,
            tools=tools,
            verbose=verbose,
            callback_manager=callback_manager,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        return super().initialize_step(task, **kwargs)

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        return super().run_step(step, task, **kwargs)

    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        return super().stream_step(step, task, **kwargs)

    def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> Coroutine[Any, Any, TaskStepOutput]:
        return super().arun_step(step, task, **kwargs)

    def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> Coroutine[Any, Any, TaskStepOutput]:
        return super().astream_step(step, task, **kwargs)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        return super().finalize_task(task, **kwargs)
