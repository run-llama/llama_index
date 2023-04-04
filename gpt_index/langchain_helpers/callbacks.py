from abc import ABC, abstractmethod
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamStdoutCallbackHandler(BaseCallbackHandler):
    """Streaming stdout callback handler.

    NOTE: this is used for demo purposes, would highly encourage you to develop your own
    callback handler.

    """

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(token, end="")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        pass
