"""ReAct multimodal agent."""

import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.agent.react_multimodal.prompts import REACT_MM_CHAT_SYSTEM_HEADER
from llama_index.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.callbacks import (
    CallbackManager,
    CBEventType,
    EventPayload,
    trace_method,
)
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    AgentChatResponse,
)
from llama_index.core.llms.types import MessageRole
from llama_index.llms.base import ChatMessage, ChatResponse
from llama_index.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.memory.types import BaseMemory
from llama_index.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.openai_utils import (
    generate_openai_multi_modal_chat_message,
)
from llama_index.objects.base import ObjectRetriever
from llama_index.tools import BaseTool, ToolOutput, adapt_to_async_tool
from llama_index.tools.types import AsyncBaseTool
from llama_index.utils import print_text

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


def add_user_step_to_reasoning(
    step: TaskStep,
    memory: BaseMemory,
    current_reasoning: List[BaseReasoningStep],
    verbose: bool = False,
) -> None:
    """Add user step to reasoning.

    Adds both text input and image input to reasoning.

    """
    # raise error if step.input is None
    if step.input is None:
        raise ValueError("Step input is None.")
    # TODO: support gemini as well. Currently just supports OpenAI

    # TODO: currently assume that you can't generate images in the loop,
    # so step_state contains the original image_docs from the task
    # (it doesn't change)
    image_docs = step.step_state["image_docs"]
    image_kwargs = step.step_state.get("image_kwargs", {})

    if "is_first" in step.step_state and step.step_state["is_first"]:
        mm_message = generate_openai_multi_modal_chat_message(
            prompt=step.input,
            role=MessageRole.USER,
            image_documents=image_docs,
            **image_kwargs,
        )
        # add to new memory
        memory.put(mm_message)
        step.step_state["is_first"] = False
    else:
        # NOTE: this is where the user specifies an intermediate step in the middle
        # TODO: don't support specifying image_docs here for now
        reasoning_step = ObservationReasoningStep(observation=step.input)
        current_reasoning.append(reasoning_step)
        if verbose:
            print(f"Added user message to memory: {step.input}")


class MultimodalReActAgentWorker(BaseAgentWorker):
    """Multimodal ReAct Agent worker.

    **NOTE**: This is a BETA feature.

    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        multi_modal_llm: MultiModalLLM,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
    ) -> None:
        self._multi_modal_llm = multi_modal_llm
        self.callback_manager = callback_manager or CallbackManager([])
        self._max_iterations = max_iterations
        self._react_chat_formatter = react_chat_formatter or ReActChatFormatter(
            system_header=REACT_MM_CHAT_SYSTEM_HEADER
        )
        self._output_parser = output_parser or ReActOutputParser()
        self._verbose = verbose

        if len(tools) > 0 and tool_retriever is not None:
            raise ValueError("Cannot specify both tools and tool_retriever")
        elif len(tools) > 0:
            self._get_tools = lambda _: tools
        elif tool_retriever is not None:
            tool_retriever_c = cast(ObjectRetriever[BaseTool], tool_retriever)
            self._get_tools = lambda message: tool_retriever_c.retrieve(message)
        else:
            self._get_tools = lambda _: []

    @classmethod
    def from_tools(
        cls,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        multi_modal_llm: Optional[MultiModalLLM] = None,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "MultimodalReActAgentWorker":
        """Convenience constructor method from set of of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgent
        """
        multi_modal_llm = multi_modal_llm or OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=1000
        )
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            multi_modal_llm=multi_modal_llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        sources: List[ToolOutput] = []
        current_reasoning: List[BaseReasoningStep] = []
        # temporary memory for new messages
        new_memory = ChatMemoryBuffer.from_defaults()

        # validation
        if "image_docs" not in task.extra_state:
            raise ValueError("Image docs not found in task extra state.")

        # initialize task state
        task_state = {
            "sources": sources,
            "current_reasoning": current_reasoning,
            "new_memory": new_memory,
        }
        task.extra_state.update(task_state)

        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            step_state={"is_first": True, "image_docs": task.extra_state["image_docs"]},
        )

    def get_tools(self, input: str) -> List[AsyncBaseTool]:
        """Get tools."""
        return [adapt_to_async_tool(t) for t in self._get_tools(input)]

    def _extract_reasoning_step(
        self, output: ChatResponse, is_streaming: bool = False
    ) -> Tuple[str, List[BaseReasoningStep], bool]:
        """
        Extracts the reasoning step from the given output.

        This method parses the message content from the output,
        extracts the reasoning step, and determines whether the processing is
        complete. It also performs validation checks on the output and
        handles possible errors.
        """
        if output.message.content is None:
            raise ValueError("Got empty message.")
        message_content = output.message.content
        current_reasoning = []
        try:
            reasoning_step = self._output_parser.parse(message_content, is_streaming)
        except BaseException as exc:
            raise ValueError(f"Could not parse output: {message_content}") from exc
        if self._verbose:
            print_text(f"{reasoning_step.get_content()}\n", color="pink")
        current_reasoning.append(reasoning_step)

        if reasoning_step.is_done:
            return message_content, current_reasoning, True

        reasoning_step = cast(ActionReasoningStep, reasoning_step)
        if not isinstance(reasoning_step, ActionReasoningStep):
            raise ValueError(f"Expected ActionReasoningStep, got {reasoning_step}")

        return message_content, current_reasoning, False

    def _process_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict: Dict[str, AsyncBaseTool] = {
            tool.metadata.get_name(): tool for tool in tools
        }
        _, current_reasoning, is_done = self._extract_reasoning_step(
            output, is_streaming
        )

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = tool.call(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    async def _aprocess_actions(
        self,
        task: Task,
        tools: Sequence[AsyncBaseTool],
        output: ChatResponse,
        is_streaming: bool = False,
    ) -> Tuple[List[BaseReasoningStep], bool]:
        tools_dict = {tool.metadata.name: tool for tool in tools}
        _, current_reasoning, is_done = self._extract_reasoning_step(
            output, is_streaming
        )

        if is_done:
            return current_reasoning, True

        # call tool with input
        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        tool = tools_dict[reasoning_step.action]
        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={
                EventPayload.FUNCTION_CALL: reasoning_step.action_input,
                EventPayload.TOOL: tool.metadata,
            },
        ) as event:
            tool_output = await tool.acall(**reasoning_step.action_input)
            event.on_end(payload={EventPayload.FUNCTION_OUTPUT: str(tool_output)})

        task.extra_state["sources"].append(tool_output)

        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
        if self._verbose:
            print_text(f"{observation_step.get_content()}\n", color="blue")
        return current_reasoning, False

    def _get_response(
        self,
        current_reasoning: List[BaseReasoningStep],
        sources: List[ToolOutput],
    ) -> AgentChatResponse:
        """Get response from reasoning steps."""
        if len(current_reasoning) == 0:
            raise ValueError("No reasoning steps were taken.")
        elif len(current_reasoning) == self._max_iterations:
            raise ValueError("Reached max iterations.")

        if isinstance(current_reasoning[-1], ResponseReasoningStep):
            response_step = cast(ResponseReasoningStep, current_reasoning[-1])
            response_str = response_step.response
        else:
            response_str = current_reasoning[-1].get_content()

        # TODO: add sources from reasoning steps
        return AgentChatResponse(response=response_str, sources=sources)

    def _get_task_step_response(
        self, agent_response: AGENT_CHAT_RESPONSE_TYPE, step: TaskStep, is_done: bool
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
            output=agent_response,
            task_step=step,
            is_last=is_done,
            next_steps=new_steps,
        )

    def _run_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        # This is either not None on the first step or if the user specifies
        # an intermediate step in the middle
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )

        # send prompt
        chat_response = self._multi_modal_llm.chat(input_chat)
        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = self._process_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)

    async def _arun_step(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        if step.input is not None:
            add_user_step_to_reasoning(
                step,
                task.extra_state["new_memory"],
                task.extra_state["current_reasoning"],
                verbose=self._verbose,
            )
        # TODO: see if we want to do step-based inputs
        tools = self.get_tools(task.input)

        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )
        # send prompt
        chat_response = await self._multi_modal_llm.achat(input_chat)
        # given react prompt outputs, call tools or return response
        reasoning_steps, is_done = await self._aprocess_actions(
            task, tools, output=chat_response
        )
        task.extra_state["current_reasoning"].extend(reasoning_steps)
        agent_response = self._get_response(
            task.extra_state["current_reasoning"], task.extra_state["sources"]
        )
        if is_done:
            task.extra_state["new_memory"].put(
                ChatMessage(content=agent_response.response, role=MessageRole.ASSISTANT)
            )

        return self._get_task_step_response(agent_response, step, is_done)

    def _run_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        raise NotImplementedError("Stream step not implemented yet.")

    async def _arun_step_stream(
        self,
        step: TaskStep,
        task: Task,
    ) -> TaskStepOutput:
        """Run step."""
        raise NotImplementedError("Stream step not implemented yet.")

    @trace_method("run_step")
    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step."""
        return self._run_step(step, task)

    @trace_method("run_step")
    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async)."""
        return await self._arun_step(step, task)

    @trace_method("run_step")
    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        """Run step (stream)."""
        # TODO: figure out if we need a different type for TaskStepOutput
        return self._run_step_stream(step, task)

    @trace_method("run_step")
    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        """Run step (async stream)."""
        return await self._arun_step_stream(step, task)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        """Finalize task, after all the steps are completed."""
        # add new messages to memory
        task.memory.set(task.memory.get() + task.extra_state["new_memory"].get_all())
        # reset new memory
        task.extra_state["new_memory"].reset()
