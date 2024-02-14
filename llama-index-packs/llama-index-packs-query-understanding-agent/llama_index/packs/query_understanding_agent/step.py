from pydantic import PrivateAttr
from typing import Any, Dict, List, Optional
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


from llama_index.core.agent import CustomSimpleAgentWorker, Task
from typing import Dict, Any, List, Tuple, Optional
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.selectors import PydanticSingleSelector
from pydantic import Field, BaseModel

from llama_index.core.llms import ChatMessage, MessageRole

DEFAULT_PROMPT_STR = """
Given previous question/response pairs, please determine if an error has occurred in the response, and suggest \
    a modified question that will not trigger the error.

Examples of modified questions:
- The question itself is modified to elicit a non-erroneous response
- The question is augmented with context that will help the downstream system better answer the question.
- The question is augmented with examples of negative responses, or other negative questions.

An error means that either an exception has triggered, or the response is completely irrelevant to the question.

Please return the evaluation of the response in the following JSON format.

"""


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

    has_error: bool = Field(..., description="Whether the response has an error")
    requires_human_input: bool = Field(
        ...,
        description="Whether the response needs human input, if human input is required, it means it is not erroneous",
    )
    clarifying_question: str = Field(
        ..., description="The suggested new question, if human input is required"
    )
    explanation: str = Field(
        ...,
        description=(
            "The explanation for the error as well as for the clarifying question."
            "Can include the direct stack trace as well."
        ),
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
        self._router_query_engine = RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(),
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

    def _run_step(
        self, state: Dict[str, Any], task: Task, input: Optional[str] = None
    ) -> Tuple[Dict, bool]:
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
        chat_prompt_tmpl = get_chat_prompt_template(
            self.prompt_str, state["current_reasoning"]
        )
        llm_program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(output_cls=ResponseEval),
            prompt=chat_prompt_tmpl,
            llm=self.llm,
        )

        # run program, look at the result
        response_eval = llm_program(query_str=new_input, response_str=str(response))

        if self.verbose:
            print(f"> Question: {new_input}")
            print(f"> Response: {response}")
            print(f"> Response eval: {response_eval.dict()}")

        # return response
        if response_eval.requires_human_input:
            return (
                {
                    "response": response,
                    "type": "requires_human_input",
                    "clarifying_question": str(response_eval.clarifying_question),
                    "has_error": response_eval.has_error,
                    "explanation": response_eval.explanation,
                },
                True,
            )

        return {"response": response}, not response_eval.has_error

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        output, is_done = self._run_step(step.step_state, task, input=step.input)
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
