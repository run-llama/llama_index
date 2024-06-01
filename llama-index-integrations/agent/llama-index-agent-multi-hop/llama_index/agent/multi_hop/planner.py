"""MultiHop Planner Agent."""

import logging
import uuid
import re

from llama_index.core.agent.runner.planner import (
    SubTask,
    Plan,
    PlannerAgentState,
    BasePlanningAgentRunner,
)


from typing import Any, List, Optional, Sequence, Tuple, Type, TypeAlias

from llama_index.core.bridge.pydantic import BaseModel, Field, validator, create_model
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
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


class MultiHopPlannerAgent(BasePlanningAgentRunner):
    """MultiHop Planner Agent Runner."""

    def __init__(
        self,
        llm: LLM,
        retrievers: List[BaseRetriever],
        tools: Optional[Sequence[BaseTool]] = None,
        structured_context: Optional[StructuredContext] = None,
        state: Optional[PlannerAgentState] = None,
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
        self.state = state or PlannerAgentState()
        self._data_extraction_agent_worker = (
            DataExtractionAgentWorker.from_tools_and_retrievers(
                tools=tools, retrievers=retrievers
            )
        )

    def _create_sub_tasks_from_data_requirements(
        self, data_requirements: DataRequirements
    ) -> List[SubTask]:
        """Create SubTasks from a DataRequirements object."""
        return [
            SubTask(
                name=field,
                input="Extract this data field.",
                expected_output=field_desc,
                dependencies=[],
            )
            for field, field_desc in zip(
                data_requirements.data_field_names,
                data_requirements.data_field_descriptions,
            )
        ]

    def create_plan(self, input: str, **kwargs: Any) -> str:
        """Create plan. Returns the plan id."""
        # generate structured data model to get data requirements based on input
        data_requirements: DataRequirements = self.llm.structured_predict(
            DataRequirements,
            PromptTemplate(DATA_REQUIREMENTS_PROMPT_TEMPLATE),
            query=input,
        )

        # generate sub-task for each data requirement
        data_extraction_sub_tasks = self._create_sub_tasks_from_data_requirements(
            data_requirements
        )

        # merge data extraction results sub-task
        merge_sub_task = SubTask(
            name="merge_data_extractions",
            input="Use the provided data to fill in the StructuredContext data class.",
            expected_output="A StructuredContext object.",
            dependencies=data_requirements.data_field_names,
        )

        # final structured context-augmentation query response task
        query_response_task = SubTask(
            name="query_response_tasks",
            input=input,
            expected_output="Response to the query.",
            dependencies=[
                *data_requirements.data_field_names,
                "merge_data_extractions",
            ],
        )

        # plan
        plan = Plan(
            sub_tasks=[*data_extraction_sub_tasks, merge_sub_task, query_response_task]
        )

        if self.verbose:
            print(f"=== Initial plan ===")
            for sub_task in plan.sub_tasks:
                print(
                    f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                )

        plan_id = str(uuid.uuid4())
        self.state.plan_dict[plan_id] = plan

        for sub_task in plan.sub_tasks:
            self.create_task(sub_task.input, task_id=sub_task.name)

        return plan_id

    async def acreate_plan(self, input: str, **kwargs: Any) -> str:
        """Create plan (async). Returns the plan id."""
        # generate structured data model to get data requirements based on input
        data_requirements: DataRequirements = await self.llm.astructured_predict(
            DataRequirements,
            PromptTemplate(DATA_REQUIREMENTS_PROMPT_TEMPLATE),
            query=input,
        )

        # generate sub-task for each data requirement
        data_extraction_sub_tasks = self._create_sub_tasks_from_data_requirements(
            data_requirements
        )

        # merge data extraction results sub-task
        merge_sub_task = SubTask(
            name="merge_data_extractions",
            input="Use the provided data to fill in the StructuredContext data class.",
            expected_output="A StructuredContext object.",
            dependencies=data_requirements.data_field_names,
        )

        # final structured context-augmentation query response task
        query_response_task = SubTask(
            name="query_response_tasks",
            input=input,
            expected_output="Response to the query.",
            dependencies=[
                *data_requirements.data_field_names,
                "merge_data_extractions",
            ],
        )

        # plan
        plan = Plan(
            sub_tasks=[*data_extraction_sub_tasks, merge_sub_task, query_response_task]
        )

        if self.verbose:
            print(f"=== Initial plan ===")
            for sub_task in plan.sub_tasks:
                print(
                    f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                )

        plan_id = str(uuid.uuid4())
        self.state.plan_dict[plan_id] = plan

        for sub_task in plan.sub_tasks:
            self.create_task(sub_task.input, task_id=sub_task.name)

        return plan_id
