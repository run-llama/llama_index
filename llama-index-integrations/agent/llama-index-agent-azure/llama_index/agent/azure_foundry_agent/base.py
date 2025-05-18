from typing import List, Optional
import time
from azure.ai.agents.models import (
    ThreadRun,
    ToolSet,
    ToolOutput
)
from llama_index.core.agent.types import BaseAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.core.callbacks import CallbackManager, trace_method, CBEventType, EventPayload
 
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

class AzureFoundryAgent(BaseAgent):
    def __init__(
        self,
        endpoint: str,
        model: str = "gpt-4o-mini",
        name: str = "azure-agent",
        instructions: str = "You are a helpful agent",
        thread_id: Optional[str] = None,
        verbose: bool = False,
        run_retrieve_sleep_time: float = 1.0,
        callback_manager: Optional[CallbackManager] = None,
        toolset: Optional[ToolSet] = None,
    ):
        """Initialize an AzureFoundryAgent.

        Args:
            endpoint: The Azure AI Project endpoint.
            model: The model to use for the agent (e.g., "gpt-4o-mini").
            name: The name of the agent.
            instructions: Instructions for the agent.
            thread_id: An optional existing thread ID to continue a conversation.
                If None, a new thread will be created.
            verbose: Whether to print verbose output.
            run_retrieve_sleep_time: Time in seconds to sleep between polling
                for run status.
            callback_manager: An optional CallbackManager instance.
            toolset: An optional ToolSet for the agent.
        """
        self._endpoint = endpoint
        self._model = model
        self._name = name
        self._instructions = instructions
        self._verbose = verbose
        self._run_retrieve_sleep_time = run_retrieve_sleep_time
        self.callback_manager = callback_manager or CallbackManager([])
        self._toolset = toolset
        
        
        self._project_client = AIProjectClient(
            endpoint=self._endpoint,
            credential=DefaultAzureCredential()
        )
       
        # Create or use thread
        if thread_id is not None:
            self._thread_id = thread_id
        else:
            thread = self._project_client.agents.threads.create()
            self._thread_id = thread.id 
            
        if self._verbose:
            if self._toolset:
                print(f"AzureFoundryAgent initialized with provided toolset.")
            else:
                print("AzureFoundryAgent initialized without tools.")
            
        self._agent = None  # Will be created on first chat
        self._run_id: Optional[str] = None

    def _ensure_agent(self):
        """Ensure the agent is created using the Azure AI Agents API."""
        if self._agent is None:
            if self._verbose:
                if self._toolset:
                    print(f"AzureFoundryAgent creating agent with provided toolset.")
                else:
                    print("AzureFoundryAgent creating agent without tools.")
            self._agent = self._project_client.agents.create_agent(
                model=self._model,
                name=self._name,
                instructions=self._instructions,
                toolset=self._toolset,
            )
            if self._verbose:
                print(f"Created agent, ID: {self._agent.id}")

    def _run_function_calling(self, run_details: ThreadRun) -> Optional[List[ToolOutput]]:
        """
        Handles the 'requires_action' state by executing tool calls and submitting outputs.
        Returns the list of tool outputs submitted, or None if no action was taken or applicable.
        """
        if not (run_details.required_action and run_details.required_action.type == "submit_tool_outputs"):
            if self._verbose:
                print("Run does not require tool submission or no action defined.")
            return None

        # Ensure _run_id is set, as it's required for submitting tool outputs
        assert self._run_id is not None, "_run_id cannot be None when submitting tool outputs."

        # Safely access submit_tool_outputs and then tool_calls
        submit_tool_outputs_details = getattr(run_details.required_action, 'submit_tool_outputs', None)
        if not submit_tool_outputs_details:
            if self._verbose:
                print("submit_tool_outputs details are missing in required_action.")
            return None
        
        tool_calls = getattr(submit_tool_outputs_details, 'tool_calls', None)

        if not (self._toolset and tool_calls):
            if self._verbose:
                print("Toolset not available or no tool calls to execute.")
            return None

        if self._verbose:
            print(f"Executing tool calls: {tool_calls}")
        
        tool_outputs_raw = self._toolset.execute_tool_calls(tool_calls)
        
        # Ensure tool_outputs is in the correct format for submission
        # This requires converting List[Dict[str, Any]] to List[ToolOutput]
        tool_outputs: List[ToolOutput] = []
        if isinstance(tool_outputs_raw, list):
            for item in tool_outputs_raw:
                if isinstance(item, dict) and "tool_call_id" in item and "output" in item:
                    tool_outputs.append(ToolOutput(tool_call_id=item["tool_call_id"], output=item["output"]))
                elif isinstance(item, ToolOutput): # If it's already a ToolOutput object
                    tool_outputs.append(item)
                else:
                    # Handle unexpected item type if necessary, or raise an error
                    if self._verbose:
                        print(f"Warning: Unexpected item type in tool_outputs_raw: {type(item)}")
        else:
            if self._verbose:
                print(f"Warning: tool_outputs_raw is not a list: {type(tool_outputs_raw)}")

        if self._verbose:
            print(f"Formatted Tool outputs for submission: {tool_outputs}")
        
        self._project_client.agents.runs.submit_tool_outputs(
            thread_id=self._thread_id,
            run_id=self._run_id, 
            tool_outputs=tool_outputs
        )
        if self._verbose:
            print("Successfully submitted tool outputs.")
        return tool_outputs

    def run_agent(self) -> ThreadRun:
        """
        Run the agent and poll until completion. Returns the run object.
        Uses the new Azure AI Agents API.
        """
        self._ensure_agent()
        if self._agent is None or self._agent.id is None:
            raise ValueError("Agent not created or agent ID is missing before running.")
 
        run_initiation: ThreadRun = self._project_client.agents.runs.create(
            thread_id=self._thread_id,
            agent_id=self._agent.id
        )
        self._run_id = run_initiation.id
 
        current_run_details = run_initiation
        
        while current_run_details.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(self._run_retrieve_sleep_time)
            # Poll the specific run using its ID and the thread ID
            current_run_details = self._project_client.agents.runs.get(
                thread_id=self._thread_id, run_id=self._run_id
            )
            if self._verbose:
                print(f"Run status: {current_run_details.status}")
            if current_run_details.status == "requires_action":
                if self._verbose:
                    print(f"Run requires action: {current_run_details.required_action}")
                
                submitted_outputs = self._run_function_calling(current_run_details)
                if submitted_outputs is not None:
                    if self._verbose:
                        print(f"Tool function calling processed. Submitted outputs: {submitted_outputs}")
                    continue 
                else:
                    if self._verbose:
                        print("Tool function calling not applicable or failed to process, breaking polling loop.")
                    break

        if current_run_details.status == "failed":
            error_info = getattr(current_run_details, 'last_error', 'No additional error information.')
            raise ValueError(f"Run failed with status {current_run_details.status}. Error: {error_info}")
        
        return current_run_details

    @property
    def latest_message(self) -> Optional[ChatMessage]:
        """Get the latest assistant message in the thread."""
        messages_iterable = self._project_client.agents.messages.list(
            run_id=self._run_id,
            thread_id=self._thread_id,
            order="desc"
        )
        
        assistant_message_obj = next(
            (msg for msg in messages_iterable if getattr(msg, "role", None) == "assistant"),
            None
        )
        
        if assistant_message_obj:
            return self.from_azure_thread_message(assistant_message_obj)
        return None

    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> AgentChatResponse:
        """
        Internal chat logic for Azure AI Agents API.
        """
        self._ensure_agent()
        if not hasattr(self, '_thread_id') or self._thread_id is None:
            thread = self._project_client.agents.threads.create()
            self._thread_id = thread.id

        msg = self._project_client.agents.messages.create(
            thread_id=self._thread_id, role="user", content=message
        )
        if self._verbose:
            print(f"Created message, message ID: {msg.id}")
        run: ThreadRun = self.run_agent()
        latest_msg = self.latest_message
        # Ensure response_text is always a string
        response_text = latest_msg.content if latest_msg and latest_msg.content is not None else ""
        return AgentChatResponse(response=response_text)

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> AgentChatResponse:
        with self.callback_manager.event(
            CBEventType.AGENT_STEP,
            payload={EventPayload.MESSAGES: [message]},
        ) as e:
            chat_response = self._chat(message, chat_history, **kwargs)
            assert isinstance(chat_response, AgentChatResponse)
            e.on_end(payload={EventPayload.RESPONSE: chat_response})
        return chat_response

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None, **kwargs
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("stream_chat not implemented")

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        raise NotImplementedError("achat not implemented")

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        raise NotImplementedError("astream_chat not implemented")

    @staticmethod
    def from_azure_thread_message(thread_message: object) -> ChatMessage:
        """Convert an OpenAI/Azure thread message to ChatMessage."""
        # Assume thread_message is a ThreadMessage object (not a dict)
        # Use attribute access as per Azure SDK
        text_contents = [
            t for t in getattr(thread_message, 'content', [])
            if getattr(t, 'type', None) == 'text'
        ]
        text_content_str = " ".join([
            getattr(getattr(t, 'text', None), 'value', '') for t in text_contents
        ])
        return ChatMessage(
            role=getattr(thread_message, 'role', ''),
            content=text_content_str,
            additional_kwargs={
                "thread_message": thread_message,
                "thread_id": getattr(thread_message, 'thread_id', None),
                "assistant_id": getattr(thread_message, 'assistant_id', None),
                "id": getattr(thread_message, 'id', None),
                "metadata": getattr(thread_message, 'metadata', None),
            },
        )

    @staticmethod
    def from_azure_thread_messages(thread_messages: list) -> List[ChatMessage]:
        """Convert a list of OpenAI/Azure thread messages to ChatMessage list."""
        return [AzureFoundryAgent.from_azure_thread_message(m) for m in thread_messages]

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Return the chat history as a list of ChatMessage objects."""
        messages_iterable = self._project_client.agents.messages.list(thread_id=self._thread_id)
        return self.from_azure_thread_messages(list(messages_iterable))

    @property
    def thread_id(self) -> str:
        """The ID of the current agent thread."""
        return self._thread_id

    @property
    def project_client(self) -> AIProjectClient:
        """The Azure AI Project client instance."""
        return self._project_client

    @property
    def agent(self):
        """The underlying Azure AI Agent object."""
        return self._agent

    @property
    def last_message(self) -> Optional[ChatMessage]:
        """Get the last message in the current thread."""
        messages_iterable = self._project_client.agents.messages.list(thread_id=self._thread_id)
        messages_list = list(messages_iterable)
        if messages_list:
            return self.from_azure_thread_message(messages_list[-1])
        return None

    def reset(self) -> None:
        """Delete and create a new thread for the agent. Also resets the current run ID."""
        if self._thread_id:
            try:
                self._project_client.agents.threads.delete(thread_id=self._thread_id)
                if self._verbose:
                    print(f"Successfully deleted old thread: {self._thread_id}")
            except Exception as e:
                if self._verbose:
                    print(f"Failed to delete old thread {self._thread_id}: {e}. Proceeding to create a new one.")
        
        thread = self._project_client.agents.threads.create()
        self._thread_id = thread.id
        self._run_id = None  # Reset the run ID
        if self._verbose:
            print(f"Agent reset. New thread ID: {self._thread_id}")
