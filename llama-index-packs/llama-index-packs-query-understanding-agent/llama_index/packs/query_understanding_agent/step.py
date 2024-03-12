from llama_index.core.bridge.pydantic import Field, BaseModel, PrivateAttr
from llama_index.core import PromptTemplate
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import uuid

from llama_index.core.agent.types import (
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.agent.custom.simple import CustomSimpleAgentWorker
from llama_index.core.callbacks import (
    trace_method,
)
from llama_index.core.chat_engine.types import AgentChatResponse


from llama_index.core.agent import CustomSimpleAgentWorker, Task
from typing import Dict, Any, List, Tuple, Optional
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.prompts import ChatPromptTemplate

from llama_index.core.llms import ChatMessage, MessageRole

DEFAULT_PROMPT_STR = """
Given the question, the tools provided and response, please determine if the question is clear enough to provide the response (with the help of the tools).
If the question is not clear enough, provide a clarifying_question for user to clarify. Provide a clarifying question that only need plan text answers. DO NOT ASK the user a similar question to the one the user asks you.
If the response is sufficient, then return has_error: false, and requires_human_input: false. No need to make the response perfect, as long as it is sufficient.

Given the questions and tools, here are several ways to clarify questions:
- If there are multiple tools for each timeframe, then if the timeframe is not specified in the question, we need to ask to be clear on the timeframe.
- Be clear on which subjects the user is referring to, ask which subject the user is referring to if there are a tool for each subject

Example 1:
Tools:
A useful tool for financial documents for uber in 2022
A useful tool for financial documents for lyft in 2022

Question: What is the company's financial stats in 2022?
Response: Uber's financial stats is 20k in the year 2022

Answer:
{{"requires_human_input": true, "has_error": true, "clarifying_question": "Which company are you referring to?", "explanation": "Given the tools and the question, it is not clear which company the user is referring to"}}

Example 2:
Tools:
A useful tool for financial documents for uber in 2022
A useful tool for financial documents for lyft in 2022

Question: What is the Uber's financial stats in 2022?
Response: Uber's financial stats is 20k in the year 2022

Answer:
{{"requires_human_input": false, "has_error": false, "clarifying_question": "", "explanation": "It is quite clear that the user is referring to uber and the year 2022"}}

Tools:
{tools}

Question: {query_str}
Response: {response_str}

Please return the evaluation of the response in the following JSON format.
Answer:
"""


@dataclass
class AgentChatComplexResponse(AgentChatResponse):
    """Agent chat response with metadata."""

    output: Dict[str, Any] = field(default_factory=dict)


def get_chat_prompt_template(
    system_prompt: str, current_reasoning: Tuple[str, str]
) -> ChatPromptTemplate:
    system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
    messages = [system_msg]
    for raw_msg in current_reasoning:
        if raw_msg[0] == "user":
            messages.append(ChatMessage(role=MessageRole.USER, content=raw_msg[1]))
        else:
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=raw_msg[1]))
    return ChatPromptTemplate(message_templates=messages)


class ResponseEval(BaseModel):
    """Evaluation of whether the response has an error."""

    clarifying_question: str = Field(
        ..., description="The clarifying question, if human input is required"
    )
    explanation: str = Field(
        ...,
        description=(
            "The explanation for the error OR for the clarifying question."
            "Can include the direct stack trace as well."
        ),
    )
    has_error: bool = Field(..., description="Whether the response has an error")
    requires_human_input: bool = Field(
        ...,
        description="Whether the response needs human input. If true, the clarifying question should be provided.",
    )


class HumanInputRequiredException(Exception):
    """Exception raised when human input is required."""

    def __init__(
        self,
        message="Human input is required",
        task_id: Optional[str] = None,
        step: TaskStep = None,
    ):
        self.message = message
        self.task_id = task_id
        self.step = step
        super().__init__(self.message)


class QueryUnderstandingAgentWorker(CustomSimpleAgentWorker):
    """Agent worker that adds a retry layer on top of a router.

    Continues iterating until there's no errors / task is done.

    """

    prompt_str: str = Field(default=DEFAULT_PROMPT_STR)
    max_iterations: int = Field(default=10)

    _router_query_engine: RouterQueryEngine = PrivateAttr()

    def __init__(self, tools: List[BaseTool], **kwargs: Any) -> None:
        """Init params."""
        # validate that all tools are query engine tools
        for tool in tools:
            if not isinstance(tool, QueryEngineTool):
                raise ValueError(
                    f"Tool {tool.metadata.name} is not a query engine tool."
                )
        self._router_query_engine = RouterQueryEngine.from_defaults(
            llm=kwargs.get("llm"),
            select_multi=False,
            query_engine_tools=tools,
            verbose=kwargs.get("verbose", False),
        )
        super().__init__(
            tools=tools,
            **kwargs,
        )

    def _initialize_state(self, task: Task, **kwargs: Any) -> Dict[str, Any]:
        """Initialize state."""
        return {"count": 0, "current_reasoning": []}

    def _run_llm_program(self, query_str, response_str, tools):
        for _ in range(3):
            try:
                return self.llm.structured_predict(
                    ResponseEval,
                    PromptTemplate(self.prompt_str),
                    query_str=query_str,
                    response_str=str(response_str),
                    tools=tools,
                )
            except Exception as e:
                print(f"Attempt failed with error: {e}")
                continue
        raise Exception("Failed to run LLM program after 3 attempts")

    def _run_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[AgentChatComplexResponse, bool]:
        """Run step.

        Returns:
            Tuple of (agent_response, is_done)

        """
        if input is not None:
            # if input is specified, override input
            new_input = input
        elif "new_input" not in state:
            new_input = task.input
        else:
            new_input = state["new_input"]["text"]

        if self.verbose:
            print(f"> Current Input: {new_input}")

        # first run router query engine
        response = self._router_query_engine.query(new_input)

        # append to current reasoning
        state["current_reasoning"].extend(
            [("user", new_input), ("assistant", str(response))]
        )

        # Then, check for errors
        # dynamically create pydantic program for structured output extraction based on template
        tools = "\n".join([a.description for a in self._router_query_engine._metadatas])
        response_eval = self._run_llm_program(
            query_str=new_input, response_str=str(response), tools=tools
        )

        if self.verbose:
            print(f"> Question: {new_input}")
            print(f"> Response: {response}")
            print(f"> Response eval: {response_eval.dict()}")

        # return response
        if response_eval.requires_human_input:
            return (
                AgentChatComplexResponse(
                    response=response,
                    output={
                        "type": "requires_human_input",
                        "clarifying_question": str(response_eval.clarifying_question),
                        "has_error": response_eval.has_error,
                        "explanation": response_eval.explanation,
                    },
                ),
                True,
            )

        return AgentChatComplexResponse(response=response), not response_eval.has_error

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        output, is_done = self._run_step(step.step_state, task, input=step.input)
        if output.output and output.output["type"] == "requires_human_input":
            raise HumanInputRequiredException(
                message=output.output["clarifying_question"],
                task_id=task.task_id,
                step=step,
            )

        response = self._get_task_step_response(output, step, is_done)
        # sync step state with task state
        task.extra_state.update(step.step_state)
        return response

    def _get_task_step_response(
        self,
        output: Dict,
        step: TaskStep,
        is_done: bool,
    ) -> TaskStepOutput:
        """Get task step response."""
        if is_done:
            new_steps = []
        else:
            new_steps = [
                step.get_next_step(
                    step_id=str(uuid.uuid4()),
                    # NOTE: input is unused
                    input=None,
                )
            ]

        return TaskStepOutput(
            output=output,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _finalize_task(self, state: Dict[str, Any], **kwargs) -> None:
        """Finalize task."""
        # nothing to finalize here
        # this is usually if you want to modify any sort of
        # internal state beyond what is set in `_initialize_state`
