"""
By integrating sandboxing, the spoke operator, and the spoke output parser with an LLM, memory, and app, we can build a standard spoke. We demonstrate the integration of these components below.
"""

from llama_index.core.llms.llm import LLM

from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer

from llama_index.core.tools import BaseTool
from llama_index.core.settings import Settings

from .tool_importer import create_message_spoke_tool, create_function_placeholder
from .spoke_operator import SpokeOperator
from .spoke_parser import SpokeOutputParser
from .sandbox import set_mem_limit, drop_perms

from typing import Sequence, List, Optional

from llama_index.core.agent import ReActAgent

from llama_index.core.base.llms.types import ChatMessage


class Spoke:
    """
    A class representing a spoke that integrates various components such as
    sandboxing, spoke operator, and spoke output parser with an LLM, memory, and app.

    Attributes:
        tools (Sequence[BaseTool]): A sequence of tools (apps) available to the spoke. Typically, just one app.
        collab_functions (Sequence[str]): A sequence of collaborative functions.
        llm (LLM): A large language model for the spoke.
        memory (ChatMemoryBuffer): The chat memory buffer.
        spoke_operator (SpokeOperator): The spoke operator instance.
        spoke_output_parser (SpokeOutputParser): The spoke output parser instance.
        spoke_agent (ReActAgent): The ReAct agent for the spoke.
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        collab_functions: Sequence[str],
        llm: LLM = None,
        memory: ChatMemoryBuffer = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the Spoke class with tools, collaborative functions, LLM, and memory.

        Args:
            tools (Sequence[BaseTool]): A sequence of tools (apps) available to the spoke. Typically, just one app.
            collab_functions (Sequence[str]): A sequence of collaborative functions.
            llm (LLM, optional): A large language model for the spoke. Defaults to None.
            memory (ChatMemoryBuffer, optional): The chat memory buffer. Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to True.
        """
        self.tools = tools
        if self.tools:
            self.tool_name = tools[0].metadata.name
        else:
            self.tool_name = ""

        self.collab_functions = collab_functions
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=[], llm=self.llm
        )

        # Set up spoke operator
        self.spoke_operator = SpokeOperator(self.collab_functions)

        # Create a placeholder for each collabortive functionality
        func_placeholders = create_function_placeholder(self.collab_functions)

        # Set the tool and collabortive functionality list
        tool_functionality_list = (
            self.tools + func_placeholders + [create_message_spoke_tool()]
        )

        # Set up the spoke output parser
        self.spoke_output_parser = SpokeOutputParser(
            functionality_list=self.collab_functions, spoke_operator=self.spoke_operator
        )
        # Set up the spoke agent
        self.spoke_agent = ReActAgent.from_tools(
            tools=tool_functionality_list,
            llm=self.llm,
            memory=self.memory,
            output_parser=self.spoke_output_parser,
            verbose=verbose,
        )

    def chat(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """
        Perform a chat interaction with the spoke agent.

        Args:
            query (str): The query to be processed.
            chat_history (Optional[List[ChatMessage]], optional): The chat history. Defaults to None.

        Returns:
            str: The response from the spoke agent.
        """
        return self.spoke_agent.chat(query, chat_history=chat_history)

    async def achat(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """
        Perform an asynchronous chat interaction with the spoke agent.

        Args:
            query (str): The query to be processed.
            chat_history (Optional[List[ChatMessage]], optional): The chat history. Defaults to None.

        Returns:
            str: The response from the spoke agent.
        """
        return await self.spoke_agent.achat(query, chat_history=chat_history)

    def stream_chat(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """
        Perform a streaming chat interaction with the spoke agent.

        Args:
            query (str): The query to be processed.
            chat_history (Optional[List[ChatMessage]], optional): The chat history. Defaults to None.

        Returns:
            str: The final response from the spoke agent.
        """
        stream_response = self.spoke_agent.stream_chat(query, chat_history=chat_history)
        final_response = ""
        for response in stream_response.chat_stream:
            final_response += response.message.content
        return final_response

    async def astream_chat(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """
        Perform an asynchronous streaming chat interaction with the spoke agent.

        Args:
            query (str): The query to be processed.
            chat_history (Optional[List[ChatMessage]], optional): The chat history. Defaults to None.

        Returns:
            str: The final response from the spoke agent.
        """
        stream_response = await self.spoke_agent.astream_chat(
            query, chat_history=chat_history
        )
        final_response = ""
        async for response in stream_response.chat_stream:
            final_response += response.message.content
        return final_response

    def run_process(
        self, child_sock, request, spoke_id, chat_history=None, chat_method="chat"
    ):
        """
        Run the process for handling a request, setting up sandboxing, and managing chat interactions.

        Args:
            child_sock (Socket): The socket for communication with the hub.
            request (str): The request to be processed.
            spoke_id (str): The identifier for the spoke.
            chat_history (Optional[List[ChatMessage]], optional): The chat history. Defaults to None.
            chat_method (str, optional): The chat method to be used. Defaults to "chat".
        """
        # Set seccomp and setrlimit
        set_mem_limit()
        drop_perms()

        self.spoke_operator.spoke_id = spoke_id
        self.spoke_operator.child_sock = child_sock
        query = self.spoke_operator.parse_request(request)

        if chat_method == "chat":
            results = self.chat(query, chat_history)
        elif chat_method == "achat":
            results = self.achat(query, chat_history)
        elif chat_method == "stream_chat":
            results = self.stream_chat(query, chat_history)
        elif chat_method == "astream_chat":
            results = self.astream_chat(query, chat_history)
        else:
            results = "The chat method is not supported. Please use one of the following: chat, achat, stream_chat, astream_chat."

        self.spoke_operator.return_response(str(results))
