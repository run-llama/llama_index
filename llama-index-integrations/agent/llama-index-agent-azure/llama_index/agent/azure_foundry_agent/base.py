import asyncio
import json
from typing import List, Sequence, Optional

from azure.ai.agents.models import (
    MessageInputContentBlock,
    SubmitToolOutputsAction,
    RequiredFunctionToolCall,
    FunctionTool,
    ToolSet,
)
from azure.ai.agents.models._models import ThreadRun
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult
from llama_index.core.llms import ChatMessage, MockLLM
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool, ToolSelection
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool as LLamaIndexFunctionTool
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock


class AzureFoundryAgent(BaseWorkflowAgent):
    """
    Workflow-compatible Azure Foundry Agent for multi-agent orchestration.
    Inherits from BaseWorkflowAgent and SingleAgentRunnerMixin.
    Implements async methods for workflow integration using the async Azure SDK.
    """

    def __init__(
        self,
        endpoint: str,
        model: str = "gpt-4o-mini",
        name: str = "azure-agent",
        instructions: str = "You are a helpful agent",
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_retrieve_sleep_time: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, llm=MockLLM(), **kwargs)
        self._endpoint = endpoint
        self._model = model
        self._instructions = instructions
        self._run_retrieve_sleep_time = run_retrieve_sleep_time
        self._thread_id = thread_id
        self._agent_id = agent_id
        self._agent = None
        self._run_id = None
        self._credential = DefaultAzureCredential()
        self._client = AIProjectClient(endpoint=endpoint, credential=self._credential)
        self._verbose = verbose
        # self.tools = tools if tools is not None else []
        self._toolset = ToolSet()

    async def _ensure_agent(self, tools: Sequence[AsyncBaseTool]) -> None:
        if self._agent is None:
            if self._agent_id is not None:
                if self._verbose:
                    print(
                        f"[AzureFoundryWorkflowAgent] Fetching existing agent with id={self._agent_id}"
                    )
                self._agent = await self._client.agents.get_agent(self._agent_id)
            else:
                if self._verbose:
                    print(
                        f"[AzureFoundryWorkflowAgent] Creating new agent with model={self._model}, name={self.name}"
                    )

                func_tools = []
                for t in tools or []:
                    if isinstance(t, LLamaIndexFunctionTool):
                        func_tools.append(t.fn)
                if func_tools:
                    self._toolset.add(FunctionTool(functions=set(func_tools)))

                self._agent = await self._client.agents.create_agent(
                    model=self._model,
                    name=self.name,
                    instructions=self._instructions,
                    toolset=self._toolset,
                )
                self._agent_id = self._agent.id
                if self._verbose:
                    print(
                        f"[AzureFoundryWorkflowAgent] Created agent with id={self._agent_id}"
                    )
        if self._thread_id is None:
            if self._verbose:
                print(f"[AzureFoundryWorkflowAgent] Creating new thread.")
            thread = await self._client.agents.threads.create()
            self._thread_id = thread.id
            if self._verbose:
                print(
                    f"[AzureFoundryWorkflowAgent] Created thread with id={self._thread_id}"
                )

    def _llama_to_azure_content_blocks(
        self, chat_messages: List[ChatMessage]
    ) -> list[MessageInputContentBlock]:
        """
        Internal: Convert a list of LlamaIndex ChatMessage to a list of Azure MessageInputContentBlock.
        Supports text and image blocks. Extend as needed for audio/document.
        """
        from azure.ai.agents.models import (
            MessageInputTextBlock,
            MessageInputImageFileBlock,
            MessageInputImageUrlBlock,
            MessageImageFileParam,
            MessageImageUrlParam,
        )

        azure_blocks: list[MessageInputContentBlock] = []
        for msg in chat_messages:
            for block in getattr(msg, "blocks", []):
                if isinstance(block, TextBlock):
                    azure_blocks.append(MessageInputTextBlock(text=block.text))
                elif isinstance(block, ImageBlock):
                    if block.path or block.image:
                        file_id = str(block.path) if block.path else None
                        if file_id:
                            azure_blocks.append(
                                MessageInputImageFileBlock(
                                    image_file=MessageImageFileParam(
                                        file_id=file_id, detail=block.detail
                                    )
                                )
                            )
                    elif block.url:
                        azure_blocks.append(
                            MessageInputImageUrlBlock(
                                image_url=MessageImageUrlParam(
                                    url=str(block.url), detail=block.detail
                                )
                            )
                        )
                else:
                    raise ValueError(f"Unsupported block type: {type(block)}")

        return azure_blocks

    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """
        Take a single step with the Azure Foundry agent.
        Interacts with Azure backend and returns AgentOutput (response, tool_calls, etc).
        """
        # Convert the entire llm_input to Azure content blocks
        azure_content_blocks = (
            self._llama_to_azure_content_blocks(llm_input) if llm_input else []
        )
        await self._ensure_agent(tools=tools)
        assert self._thread_id is not None, (
            "Thread ID must be set after _ensure_agent()"
        )
        assert self._agent is not None, "Agent must be set after _ensure_agent()"

        tool_calls = []
        response_msg = None
        # Only send a user message if there is new user input
        if azure_content_blocks:
            if self._verbose:
                print(
                    f"[AzureFoundryWorkflowAgent] Sending user message blocks to thread_id={self._thread_id}"
                )
            await self._client.agents.messages.create(
                thread_id=self._thread_id, role="user", content=azure_content_blocks
            )
            if self._verbose:
                print(
                    f"[AzureFoundryWorkflowAgent] Starting run for agent_id={self._agent.id} on thread_id={self._thread_id}"
                )
            run = await self._client.agents.runs.create(
                thread_id=self._thread_id, agent_id=self._agent.id
            )
            self._run_id = run.id
            current_run = run
            while current_run.status in ["queued", "in_progress", "requires_action"]:
                await asyncio.sleep(self._run_retrieve_sleep_time)
                current_run = await self._client.agents.runs.get(
                    thread_id=self._thread_id, run_id=self._run_id
                )
                if self._verbose:
                    print(
                        f"[AzureFoundryWorkflowAgent] Run status: {current_run.status}"
                    )
                if current_run.status == "requires_action":
                    if self._verbose:
                        print(
                            f"[AzureFoundryWorkflowAgent] Run requires action: {getattr(current_run, 'required_action', None)}"
                        )
                    break

            if current_run.status == "failed":
                return AgentOutput(
                    response=ChatMessage(role="assistant", content="Run failed."),
                    tool_calls=[],
                    raw=current_run,
                    current_agent_name=self.name,
                )
            required_action = getattr(current_run, "required_action", None)
            if (
                required_action
                and getattr(required_action, "type", None) == "submit_tool_outputs"
                and isinstance(required_action, SubmitToolOutputsAction)
            ):
                submit_tool_outputs = required_action.submit_tool_outputs
                for call in getattr(submit_tool_outputs, "tool_calls", []):
                    # For function tool calls
                    if isinstance(call, RequiredFunctionToolCall):
                        function = getattr(call, "function", None)
                        tool_name = getattr(function, "name", "") if function else ""
                        arguments = (
                            getattr(function, "arguments", "{}") if function else "{}"
                        )
                        try:
                            tool_kwargs = json.loads(arguments)
                        except Exception:
                            tool_kwargs = {}
                        tool_calls.append(
                            ToolSelection(
                                tool_id=getattr(call, "id", ""),
                                tool_name=tool_name,
                                tool_kwargs=tool_kwargs,
                            )
                        )
            # Get the latest assistant message if available
            latest_msg = None
            async for msg in self._client.agents.messages.list(
                thread_id=self._thread_id, run_id=self._run_id, order="desc"
            ):
                if getattr(msg, "role", None) == "assistant" and getattr(
                    msg, "content", None
                ):
                    latest_msg = self._from_azure_thread_message(msg)
                    break
            # If no assistant message found, try to get the last assistant message in the thread
            if not latest_msg:
                async for msg in self._client.agents.messages.list(
                    thread_id=self._thread_id, order="desc"
                ):
                    if getattr(msg, "role", None) == "assistant" and getattr(
                        msg, "content", None
                    ):
                        latest_msg = self._from_azure_thread_message(msg)
                        break
            response_msg = (
                latest_msg
                if latest_msg
                else ChatMessage(role="assistant", content="No response from agent.")
            )
        else:
            # No new user input: fetch the latest assistant message after tool call resolution
            latest_msg = None
            async for msg in self._client.agents.messages.list(
                thread_id=self._thread_id, order="desc"
            ):
                if getattr(msg, "role", None) == "assistant" and getattr(
                    msg, "content", None
                ):
                    latest_msg = self._from_azure_thread_message(msg)
                    break
            response_msg = (
                latest_msg
                if latest_msg
                else ChatMessage(role="assistant", content="No response from agent.")
            )

        return AgentOutput(
            response=response_msg,
            tool_calls=tool_calls,
            raw=current_run if azure_content_blocks else None,
            current_agent_name=self.name,
        )

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        """
        Handle tool call results for Azure Foundry agent.
        Submits results to Azure backend and updates state/context as needed.
        Waits for run to reach a terminal state or another action required.
        Also appends tool call results to the scratchpad for context tracking.
        """
        # Convert ToolCallResult to Azure tool_outputs format
        tool_outputs = []
        for result in results:
            tool_outputs.append(
                {
                    "tool_call_id": result.tool_id,
                    "output": result.tool_output.content,
                }
            )
        # Submit tool outputs to Azure
        assert self._thread_id is not None, "Thread ID must be set."
        assert self._run_id is not None, "Run ID must be set."
        if self._verbose:
            print(
                f"[AzureFoundryWorkflowAgent] Submitting tool call results for run_id={self._run_id} on thread_id={self._thread_id}: {tool_outputs}"
            )
        await self._client.agents.runs.submit_tool_outputs(
            thread_id=self._thread_id, run_id=self._run_id, tool_outputs=tool_outputs
        )

        if self._verbose:
            print(
                f"[AzureFoundryWorkflowAgent] Tool outputs submitted. Waiting for run to reach terminal state or next action required..."
            )
        # Wait for run to reach a terminal state or another action required
        while True:
            run: ThreadRun = await self._client.agents.runs.get(
                thread_id=self._thread_id, run_id=self._run_id
            )
            if run.status not in ["queued", "in_progress", "requires_action"]:
                if self._verbose:
                    print(
                        f"[AzureFoundryWorkflowAgent] Run reached terminal state: {run.status}"
                    )
                # Print detailed debug info if failed
                if run.status == "failed":
                    print(
                        "[AzureFoundryWorkflowAgent][DEBUG] Run failed. Full run object:"
                    )
                    print(run)
                    # Try to print error fields if present
                    error_fields = [
                        "error",
                        "last_error",
                        "failure_reason",
                        "failure_message",
                    ]
                    for field in error_fields:
                        if hasattr(run, field):
                            print(
                                f"[AzureFoundryWorkflowAgent][DEBUG] {field}: {getattr(run, field)}"
                            )
                break
            if run.status == "requires_action":
                if self._verbose:
                    print(
                        f"[AzureFoundryWorkflowAgent] Run requires another action: {getattr(run, 'required_action', None)}"
                    )
                break
            await asyncio.sleep(self._run_retrieve_sleep_time)

        tool_message = f"A tool call was executed : {results!s}"
        await self._client.agents.messages.create(
            thread_id=self._thread_id, role="assistant", content=tool_message
        )

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        """
        Finalize the agent's execution (persist state, cleanup, etc).
        For AzureFoundryWorkflowAgent, this can be a no-op or can persist any final state if needed.
        """
        # Optionally, persist any final state to memory or context
        # For now, just return the output as-is
        return output

    async def close(self):
        """
        Close the underlying async Azure client session and credential.
        """
        if self._verbose:
            print(f"[AzureFoundryWorkflowAgent] Closing the session.")
        await self._client.close()
        await self._credential.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def _from_azure_thread_message(self, thread_message: object) -> ChatMessage:
        """
        Convert an Azure/OpenAI thread message to a LlamaIndex ChatMessage.
        Supports text and image_url content blocks for multimodal support.
        """
        from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock

        blocks = []
        for t in getattr(thread_message, "content", []):
            t_type = getattr(t, "type", None)
            if t_type == "text":
                text_val = getattr(getattr(t, "text", None), "value", "")
                blocks.append(TextBlock(text=text_val))
            elif t_type == "image_url":
                url_val = getattr(getattr(t, "image_url", None), "url", None)
                detail_val = getattr(getattr(t, "image_url", None), "detail", None)
                if url_val:
                    blocks.append(ImageBlock(url=url_val, detail=detail_val))
        # Compose content string for backward compatibility (concatenate text blocks)
        content_str = " ".join([b.text for b in blocks if hasattr(b, "text")])
        return ChatMessage(
            role=getattr(thread_message, "role", ""),
            content=content_str,
            blocks=blocks,
            additional_kwargs={
                "thread_message": thread_message,
                "thread_id": getattr(thread_message, "thread_id", None),
                "assistant_id": getattr(thread_message, "assistant_id", None),
                "id": getattr(thread_message, "id", None),
                "metadata": getattr(thread_message, "metadata", None),
            },
        )
