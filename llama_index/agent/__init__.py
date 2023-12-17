# agent runner + step engine
from llama_index.agent.executor.base import AgentEngine
from llama_index.agent.executor.parallel import ParallelAgentEngine
from llama_index.agent.legacy.context_retriever_agent import ContextRetrieverOpenAIAgent
from llama_index.agent.legacy.openai_agent import OpenAIAgent as OldOpenAIAgent
from llama_index.agent.legacy.react.base import ReActAgent as OldReActAgent
from llama_index.agent.legacy.retriever_openai_agent import FnRetrieverOpenAIAgent
from llama_index.agent.openai.base import OpenAIAgent
from llama_index.agent.openai.step import OpenAIAgentStepEngine
from llama_index.agent.openai_assistant_agent import OpenAIAssistantAgent
from llama_index.agent.react.base import ReActAgent
from llama_index.agent.react.step import ReActAgentStepEngine

# for backwards compatibility
RetrieverOpenAIAgent = FnRetrieverOpenAIAgent

__all__ = [
    "AgentEngine",
    "ParallelAgentEngine",
    "OpenAIAgentStepEngine",
    "ReActAgentStepEngine",
    "OpenAIAgent",
    "ReActAgent",
    "OpenAIAssistantAgent",
    "FnRetrieverOpenAIAgent",
    "RetrieverOpenAIAgent",  # for backwards compatibility
    "ContextRetrieverOpenAIAgent",
    # legacy
    "OldOpenAIAgent",
    "OldReActAgent",
]
