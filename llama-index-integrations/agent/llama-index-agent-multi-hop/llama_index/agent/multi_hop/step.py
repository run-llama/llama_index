"""Multi-Hop agent worker."""

import logging
import uuid
import re
from typing import Any, Coroutine, List, Optional, Sequence

from llama_index.core.agent.types import (
    BaseAgentWorker,
)
from llama_index.core.base.agent.types import Task, TaskStep, TaskStepOutput
from llama_index.core.bridge.pydantic import BaseModel, Field, validator, create_model
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
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


class DataRequirements(BaseModel):
    """Data class for holding the data requirements for answering the query."""

    data_field_names: List[str] = Field(
        default_factory=list,
        description="List of data field names required to answer the query.",
    )
    data_field_descriptions: List[str] = Field(
        default_factory=list,
        description="Corresponding descriptions of each data field name.",
    )

    @validator("data_field_descriptions")
    def must_have_same_length_as_field_names(cls, v, values):
        if len(v) != len(values["data_field_names"]):
            raise ValueError("There must be a description for every data field.")
        return v

    def to_structured_context(self) -> BaseModel:
        """Generate a custom pydantic model for StructuredContext."""
        data_fields = {}
        for name, desc in zip(self.data_field_names, self.data_field_descriptions):
            # strip out punctuation
            name = re.sub(r"[^\w\s]", "", name)
            # lower and replace space with _
            name = name.lower().replace(" ", "_")
            data_fields[name] = (Optional[str], Field(default=None, description=desc))

        StructuredContext = create_model("StructuredContext", **data_fields)
        StructuredContext.__doc__ = (
            "Data class for holding data requirements to answer query"
        )
        return StructuredContext


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
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # put current history in new memory
        messages = task.memory.get(input=task.input)
        for message in messages:
            new_memory.put(message)

        # initialize task state
        task_state = {"new_memory": new_memory}
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id, step_id=str(uuid.uuid4()), input=task.input
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        # generate structured data model to get data requirements based on input

        # retrieve relevant documents from index

        # perform data extraction using retrieved context from RAG

        # perform final response synthesis
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
