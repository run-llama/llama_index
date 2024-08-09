"""Multi-Hop agent worker."""

import logging
import uuid
import re
from typing import Any, Coroutine, List, Optional, Sequence, Tuple, Type, TypeAlias

from llama_index.core.agent.types import (
    BaseAgentWorker,
)
from llama_index.core.base.agent.types import Task, TaskStep, TaskStepOutput
from llama_index.core.bridge.pydantic import BaseModel, Field, validator, create_model
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool

from llama_index.agent.multi_hop.data_extraction import DataExtractionAgentWorker

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

StructuredContext: TypeAlias = BaseModel


def get_function_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    """Get function by name."""
    name_to_tool = {tool.metadata.name: tool for tool in tools}
    if name not in name_to_tool:
        raise ValueError(f"Tool with name {name} not found")
    return name_to_tool[name]


DATA_REQUIREMENTS_PROMPT_TEMPLATE = """
You are responsible for identifying the data requirements for
sufficiently answering the query provided below. Make sure to
also include the direct object of the query itself as a data field
requirement.

{query}
"""


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

    def get_structured_context_cls(
        self,
    ) -> Tuple[Type[StructuredContext], List[Type[StructuredContext]]]:
        """Generate a custom pydantic model for StructuredContext."""
        data_fields = {}
        sub_structured_context_classes = []
        for name, desc in zip(self.data_field_names, self.data_field_descriptions):
            this_data_field = {}
            # strip out punctuation
            name = re.sub(r"[^\w\s]", "", name)
            # lower and replace space with _
            name = name.lower().replace(" ", "_")
            this_data_field[name] = (
                Optional[str],
                Field(default=None, description=desc),
            )
            data_fields.update(this_data_field)
            sub_structured_context_classes.append(create_model(""))

        StructuredContext = create_model("StructuredContext", **data_fields)
        StructuredContext.__doc__ = (
            "Data class for holding data requirements to answer query"
        )
        return StructuredContext


def _get_query_str_from_structured_context_cls(
    structured_context_cls: Type[StructuredContext],
) -> str:
    """Generate query string from a structure context class."""
    query_str = ""
    for props in structured_context_cls.schema()["properties"].values():
        query_str += f"{props['description']}\n\n"
    return query_str


class MultiHopAgentWorker(BaseAgentWorker):
    """MultiHop Agent Worker."""

    def __init__(
        self,
        llm: LLM,
        retrievers: List[BaseRetriever],
        tools: Optional[Sequence[BaseTool]] = None,
        structured_context: Optional[StructuredContext] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self.llm = llm
        self.retrievers = retrievers
        self.tools = tools
        self.verbose = verbose
        self.callback_manager = callback_manager or CallbackManager([])
        self.structured_context = structured_context
        self._data_extraction_agent_worker = (
            DataExtractionAgentWorker.from_tools_and_retrievers(
                tools=tools, retrievers=retrievers
            )
        )

    @classmethod
    def from_defaults(
        cls,
        retrievers: List[BaseRetriever],
        llm: Optional[LLM] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        structured_context: Optional[StructuredContext] = None,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> "MultiHopAgentWorker":
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        return cls(
            llm=llm,
            retrievers=retrievers,
            tools=tools,
            structured_context=structured_context,
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
        task_state = {
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"structured_context": self.structured_context},
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        structured_context = step.step_state["structured_context"]
        if structured_context is None:
            # generate structured data model to get data requirements based on input
            data_requirements: DataRequirements = self.llm.structured_predict(
                DataRequirements,
                PromptTemplate(DATA_REQUIREMENTS_PROMPT_TEMPLATE),
                query=task.input,
            )
            structured_context_cls = data_requirements.get_structured_context_cls()

            # generate sub-task for each data requirement
            sub_task_extra_state = {"structured_context_obj": structured_context_cls()}
            data_extraction_agent = self._data_extraction_agent_worker.as_agent(
                extra_state=sub_task_extra_state
            )
            data_extraction_response = data_extraction_agent.chat(
                "Perform data extraction task."
            )

            # try parsing as a structured context obj
            final_structured_context = structured_context_cls.parse_obj(
                data_extraction_response.content
            )

        # perform final response synthesis
        structured_context_augmented_query = ...
        result = self.llm.chat(structured_context_augmented_query)

        agent_response = AgentChatResponse(
            response=str(result.message),
            sources=task.extra_state["data_extraction"]["sources"],
        )

        return TaskStepOutput(
            output=agent_response,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

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
