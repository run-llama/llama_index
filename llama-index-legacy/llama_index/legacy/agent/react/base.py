"""ReAct agent.

Simple wrapper around AgentRunner + ReActAgentWorker.

For the legacy implementation see:
```python
from llama_index.legacy.agent.legacy.react.base import ReActAgent
```

"""

from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Type,
)

from llama_index.legacy.agent.react.formatter import ReActChatFormatter
from llama_index.legacy.agent.react.output_parser import ReActOutputParser
from llama_index.legacy.agent.react.step import ReActAgentWorker
from llama_index.legacy.agent.runner.base import AgentRunner
from llama_index.legacy.callbacks import (
    CallbackManager,
)
from llama_index.legacy.core.llms.types import ChatMessage
from llama_index.legacy.llms.llm import LLM
from llama_index.legacy.llms.openai import OpenAI
from llama_index.legacy.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.legacy.memory.types import BaseMemory
from llama_index.legacy.objects.base import ObjectRetriever
from llama_index.legacy.prompts.mixin import PromptMixinType
from llama_index.legacy.tools import BaseTool

DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"


class ReActAgent(AgentRunner):
    """ReAct agent.

    Subclasses AgentRunner with a ReActAgentWorker.

    For the legacy implementation see:
    ```python
    from llama_index.legacy.agent.legacy.react.base import ReActAgent
    ```

    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        context: Optional[str] = None,
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager
        if context and react_chat_formatter:
            raise ValueError("Cannot provide both context and react_chat_formatter")
        if context:
            react_chat_formatter = ReActChatFormatter.from_context(context)

        step_engine = ReActAgentWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
        )
        super().__init__(
            step_engine,
            memory=memory,
            llm=llm,
            callback_manager=callback_manager,
        )

    @classmethod
    def from_tools(
        cls,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        max_iterations: int = 10,
        react_chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> "ReActAgent":
        """Convenience constructor method from set of of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        Returns:
            ReActAgent
        """
        llm = llm or OpenAI(model=DEFAULT_MODEL_NAME)
        if callback_manager is not None:
            llm.callback_manager = callback_manager
        memory = memory or memory_cls.from_defaults(
            chat_history=chat_history or [], llm=llm
        )
        return cls(
            tools=tools or [],
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            context=context,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"agent_worker": self.agent_worker}
