"""Create LlamaIndex agents."""

from typing import Any, Optional

from llama_index.core.bridge.langchain import (
    AgentExecutor,
    AgentType,
    BaseCallbackManager,
    BaseLLM,
    initialize_agent,
)
from llama_index.core.langchain_helpers.agents.toolkits import LlamaToolkit


def create_llama_agent(
    toolkit: LlamaToolkit,
    llm: BaseLLM,
    agent: Optional[AgentType] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    agent_path: Optional[str] = None,
    agent_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given a Llama Toolkit and LLM.

    NOTE: this is a light wrapper around initialize_agent in langchain.

    Args:
        toolkit: LlamaToolkit to use.
        llm: Language model to use as the agent.
        agent: A string that specified the agent type to use. Valid options are:
            `zero-shot-react-description`
            `react-docstore`
            `self-ask-with-search`
            `conversational-react-description`
            `chat-zero-shot-react-description`,
            `chat-conversational-react-description`,
           If None and agent_path is also None, will default to
            `zero-shot-react-description`.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided. Defaults to None.
        agent_path: Path to serialized agent to use.
        agent_kwargs: Additional key word arguments to pass to the underlying agent
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """
    llama_tools = toolkit.get_tools()
    return initialize_agent(
        llama_tools,
        llm,
        agent=agent,
        callback_manager=callback_manager,
        agent_path=agent_path,
        agent_kwargs=agent_kwargs,
        **kwargs,
    )


def create_llama_chat_agent(
    toolkit: LlamaToolkit,
    llm: BaseLLM,
    callback_manager: Optional[BaseCallbackManager] = None,
    agent_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Load a chat llama agent given a Llama Toolkit and LLM.

    Args:
        toolkit: LlamaToolkit to use.
        llm: Language model to use as the agent.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided. Defaults to None.
        agent_kwargs: Additional key word arguments to pass to the underlying agent
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """
    # chat agent
    # TODO: explore chat-conversational-react-description
    agent_type = AgentType.CONVERSATIONAL_REACT_DESCRIPTION
    return create_llama_agent(
        toolkit,
        llm,
        agent=agent_type,
        callback_manager=callback_manager,
        agent_kwargs=agent_kwargs,
        **kwargs,
    )
