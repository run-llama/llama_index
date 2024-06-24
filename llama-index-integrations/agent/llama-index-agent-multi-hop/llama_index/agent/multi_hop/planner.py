"""MultiHop Planner Agent."""

import json
import logging
import uuid
import re

from typing import Any, List, Optional, Sequence, Tuple, Type, TypeAlias, Union
from llama_index.core.agent.types import (
    BaseAgentWorker,
)
from llama_index.core.agent.runner.planner import (
    SubTask,
    Plan,
    PlannerAgentState,
    BasePlanningAgentRunner,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.bridge.pydantic import BaseModel, Field, validator, create_model
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    ChatResponseMode,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import BaseTool
from llama_index.core.settings import Settings

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
You have access to an all-knowing database of facts in order to answer queries.

As such, your first step is generate a set of sub questions that you think are
necessary to answer the original query. Store your sub questions in the data class.

Next, make sure to convert each sub question into a data field requirement and store it in the data class.

Finally, convert the original query itself as a data field requirement and store it in the data class.

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
    sub_questions: List[str] = Field(
        default_factory=list, description="Sub questions to the original query."
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


class MultiHopPlannerAgent(BasePlanningAgentRunner):
    """MultiHop Planner Agent Runner."""

    def __init__(
        self,
        agent_worker: BaseAgentWorker,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        llm: Optional[LLM] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        structured_context: Optional[StructuredContext] = None,
        state: Optional[PlannerAgentState] = None,
        init_task_state_kwargs: Optional[dict] = None,
        delete_task_on_finish: bool = False,
        default_tool_choice: str = "auto",
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Init params."""
        self.agent_worker = agent_worker
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer.from_defaults(chat_history, llm=llm)
        self.tools = tools
        self.init_task_state_kwargs = init_task_state_kwargs or {}
        self.delete_task_on_finish = delete_task_on_finish
        self.default_tool_choice = default_tool_choice
        self.verbose = verbose
        self.callback_manager = callback_manager or CallbackManager([])
        self.structured_context = structured_context
        self.state = state or PlannerAgentState()

    def _create_sub_tasks_from_structured_context_cls(
        self, structured_context_cls: Type[StructuredContext]
    ) -> List[SubTask]:
        """Create data extraction subtasks from dynamic structured context class."""
        return [
            SubTask(
                name=field,
                input="Extract this data field.",
                expected_output=field_props["description"],
                dependencies=[],
            )
            for field, field_props in structured_context_cls.schema()[
                "properties"
            ].items()
        ]

    def get_next_tasks(self, plan_id: str, **kwargs: Any) -> List[str]:
        """Get next task ids for a given plan."""
        upcoming_sub_tasks = self.state.get_next_sub_tasks(plan_id)
        return [sub_task.name for sub_task in upcoming_sub_tasks]

    def mark_task_complete(self, plan_id: str, task_id: str, **kwargs: Any) -> None:
        """Mark task complete for a given plan."""
        sub_tasks_by_id = {
            sub_task.name: sub_task
            for sub_task in self.state.plan_dict[plan_id].sub_tasks
        }
        self.state.add_completed_sub_task(plan_id, sub_tasks_by_id[task_id])

    def create_plan(self, input: str, **kwargs: Any) -> str:
        """Create plan. Returns the plan id."""
        # generate structured data model to get data requirements based on input
        data_requirements: DataRequirements = self.llm.structured_predict(
            DataRequirements,
            PromptTemplate(DATA_REQUIREMENTS_PROMPT_TEMPLATE),
            query=input,
        )
        structured_context_cls = data_requirements.get_structured_context_cls()

        # generate sub-task for each data requirement
        data_extraction_sub_tasks = self._create_sub_tasks_from_structured_context_cls(
            structured_context_cls
        )

        # merge data extraction results sub-task
        merge_sub_task = SubTask(
            name="merge_data_extractions",
            input="Use the provided data to fill in the StructuredContext data class.",
            expected_output="A StructuredContext object.",
            dependencies=[*structured_context_cls.schema()["properties"].keys()],
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
            print(f"## Structured Context")
            print(json.dumps(structured_context_cls.schema(), indent=4))
            print()
            print(f"## Sub Tasks")
            for sub_task in plan.sub_tasks:
                print(
                    f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                )

        plan_id = str(uuid.uuid4())
        self.state.plan_dict[plan_id] = plan

        for sub_task in plan.sub_tasks:
            self.create_task(sub_task.input, task_id=sub_task.name)

        return plan_id

    def refine_plan(self, input: str, plan_id: str, **kwargs: Any) -> None:
        return

    def run_task(
        self,
        task_id: str,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        tool_choice: Union[str, dict] = "auto",
        **kwargs: Any,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Run a task."""
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self._run_step(
                task_id, mode=mode, tool_choice=tool_choice
            )

            if cur_step_output.is_last:
                result_output = cur_step_output
                break

            # ensure tool_choice does not cause endless loops
            tool_choice = "auto"

        return self.finalize_response(
            task_id,
            result_output,
        )

    async def acreate_plan(self, input: str, **kwargs: Any) -> str:
        """Create plan (async). Returns the plan id."""
        # generate structured data model to get data requirements based on input
        data_requirements: DataRequirements = await self.llm.astructured_predict(
            DataRequirements,
            PromptTemplate(DATA_REQUIREMENTS_PROMPT_TEMPLATE),
            query=input,
        )
        structured_context_cls = data_requirements.get_structured_context_cls()

        # generate sub-task for each data requirement
        data_extraction_sub_tasks = self._create_sub_tasks_from_structured_context_cls(
            structured_context_cls
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
            print(f"## Structured Context")
            print(json.dumps(structured_context_cls.schema(), indent=4))
            print()
            print(f"## Sub Tasks")
            for sub_task in plan.sub_tasks:
                print(
                    f"{sub_task.name}:\n{sub_task.input} -> {sub_task.expected_output}\ndeps: {sub_task.dependencies}\n\n"
                )

        plan_id = str(uuid.uuid4())
        self.state.plan_dict[plan_id] = plan

        for sub_task in plan.sub_tasks:
            self.create_task(sub_task.input, task_id=sub_task.name)

        return plan_id
