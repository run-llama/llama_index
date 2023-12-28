# agent runner + agent worker
from llama_index.agent.legacy.context_retriever_agent import ContextRetrieverOpenAIAgent
from llama_index.agent.legacy.openai_agent import OpenAIAgent as OldOpenAIAgent
from llama_index.agent.legacy.react.base import ReActAgent as OldReActAgent
from llama_index.agent.legacy.retriever_openai_agent import FnRetrieverOpenAIAgent
from llama_index.agent.openai.base import OpenAIAgent
from llama_index.agent.openai.step import OpenAIAgentWorker
from llama_index.agent.openai_assistant_agent import OpenAIAssistantAgent
from llama_index.agent.react.base import ReActAgent
from llama_index.agent.react.step import ReActAgentWorker
from llama_index.agent.runner.base import AgentRunner
from llama_index.agent.runner.parallel import ParallelAgentRunner

# for backwards compatibility
RetrieverOpenAIAgent = FnRetrieverOpenAIAgent

__all__ = [
    "AgentRunner",
    "ParallelAgentRunner",
    "OpenAIAgentWorker",
    "ReActAgentWorker",
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
