"""ReAct agent.

Simple wrapper around AgentRunner + ReActAgentWorker.

For the legacy implementation see:
```python
from llama_index.core.agent.legacy.react.base import ReActAgent
```

"""
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Type,
    Callable,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.step import ReActAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, ToolOutput
from llama_index.core.prompts.mixin import PromptMixinType


class ReActAgent(AgentRunner):
    """ReAct agent.

    Subclasses AgentRunner with a ReActAgentWorker.

    For the legacy implementation see:
    ```python
    from llama_index.core.agent.legacy.react.base import ReActAgent
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
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
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
            handle_reasoning_failure_fn=handle_reasoning_failure_fn,
        )
        super().__init__(
            step_engine,
            memory=memory,
            llm=llm,
            callback_manager=callback_manager,
            verbose=verbose,
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
        handle_reasoning_failure_fn: Optional[
            Callable[[CallbackManager, Exception], ToolOutput]
        ] = None,
        **kwargs: Any,
    ) -> "ReActAgent":
        """Convenience constructor method from set of BaseTools (Optional).

        NOTE: kwargs should have been exhausted by this point. In other words
        the various upstream components such as BaseSynthesizer (response synthesizer)
        or BaseRetriever should have picked up off their respective kwargs in their
        constructions.

        If `handle_reasoning_failure_fn` is provided, when LLM fails to follow the response templates specified in
        the System Prompt, this function will be called. This function should provide to the Agent, so that the Agent
        can have a second chance to fix its mistakes.
        To handle the exception yourself, you can provide a function that raises the `Exception`.

        Note: If you modified any response template in the System Prompt, you should override the method
        `_extract_reasoning_step` in `ReActAgentWorker`.

        Returns:
            ReActAgent
        """
        llm = llm or Settings.llm
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
            handle_reasoning_failure_fn=handle_reasoning_failure_fn,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {"agent_worker": self.agent_worker}
