"""
In SecGPT, if the hub planner determines that a user query can be addressed solely by an LLM, it utilizes a non-collaborative vanilla spoke, which operates without awareness of other system functionalities.
"""

from llama_index.core.llms.llm import LLM

from llama_index.core.settings import Settings
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent

from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage


class VanillaSpoke:
    """
    A non-collaborative vanilla spoke that operates without awareness of other system functionalities.
    It is used when a user query can be addressed solely by an LLM without requiring collaboration.
    """

    def __init__(
        self, llm: LLM = None, memory: ChatMemoryBuffer = None, verbose: bool = False
    ) -> None:
        """
        Initialize the VanillaSpoke with an LLM and optional memory.

        Args:
            llm (LLM, optional): A large language model for the spoke. Defaults to None.
            memory (ChatMemoryBuffer, optional): The chat memory buffer. Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.

        """
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=[], llm=self.llm
        )
        self.vanilla_agent = ReActAgent.from_tools(
            tools=None, llm=self.llm, memory=self.memory, verbose=verbose
        )

    def chat(
        self,
        query: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ):
        """
        Perform a chat interaction with the vanilla agent.

        Args:
            query (str): The query to be processed.
            chat_history (Optional[List[ChatMessage]], optional): The chat history. Defaults to None.

        Returns:
            str: The response from the vanilla agent.

        """
        return self.vanilla_agent.chat(query, chat_history=chat_history)
