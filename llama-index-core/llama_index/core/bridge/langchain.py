import langchain

# base language model
try:
    # # For langchain v1.x.x
    from langchain_core.language_models import BaseLanguageModel
except ImportError:
    # For langchain v0.x.x
    from langchain.base_language import BaseLanguageModel

# callback
try:
    # # For langchain v1.x.x
    from langchain_core.callbacks.base import (
        BaseCallbackHandler,
        BaseCallbackManager,
    )
    from langchain_classic.chains.prompt_selector import (
        ConditionalPromptSelector,
        is_chat_model,
    )
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.documents.base import Document
    from langchain_core.outputs import LLMResult

except ImportError:
    # For langchain v0.x.x
    from langchain.callbacks.base import (
        BaseCallbackHandler,
        BaseCallbackManager,
    )
    from langchain.chains.prompt_selector import (
        ConditionalPromptSelector,
        is_chat_model,
    )
    from langchain.chat_models.base import BaseChatModel
    from langchain.docstore.document import Document
    from langchain.schema import LLMResult

# prompts
try:
    # # For langchain v1.x.x
    from langchain_core.prompts import PromptTemplate
    from langchain_core.prompts.chat import (
        AIMessagePromptTemplate,
        BaseMessagePromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
except ImportError:
    # For langchain v0.x.x
    from langchain.prompts import PromptTemplate
    from langchain.prompts.chat import (
        AIMessagePromptTemplate,
        BaseMessagePromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )

# schema
try:
    # # For langchain v1.x.x
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
    )
except ImportError:
    # For langchain v0.x.x
    from langchain.schema import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
    )

# embeddings
try:
    # # For langchain v1.x.x
    from langchain_core.embeddings import Embeddings
    from langchain_core.prompts import BasePromptTemplate
except ImportError:
    # For langchain v0.x.x
    from langchain.schema.embeddings import Embeddings
    from langchain.schema.prompt_template import (
        BasePromptTemplate,
    )

# tools
try:
    # # For langchain v1.x.x
    from langchain_core.tools import BaseTool, StructuredTool, Tool
except ImportError:
    # For langchain v0.x.x
    from langchain.tools import BaseTool, StructuredTool, Tool

# Models
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except (ImportError, AttributeError):
        ChatOpenAI = None  # type: ignore[assignment, misc]

try:
    from langchain_fireworks import ChatFireworks
except ImportError:
    try:
        from langchain_community.chat_models import ChatFireworks
    except (ImportError, AttributeError):
        ChatFireworks = None  # type: ignore[assignment, misc]

try:
    from langchain_community.chat_models import ChatAnyscale
except (ImportError, AttributeError):
    ChatAnyscale = None  # type: ignore[assignment, misc]

try:
    from langchain_community.embeddings import (
        HuggingFaceBgeEmbeddings,
        HuggingFaceEmbeddings,
    )
except (ImportError, AttributeError):
    HuggingFaceBgeEmbeddings = None  # type: ignore[assignment, misc]
    HuggingFaceEmbeddings = None  # type: ignore[assignment, misc]

# LLMs
try:
    from langchain_community.llms import (
        AI21,
        BaseLLM,
        Cohere,
        FakeListLLM,
        OpenAI,
    )
except (ImportError, AttributeError):
    AI21 = None  # type: ignore[assignment, misc]
    BaseLLM = None  # type: ignore[assignment, misc]
    Cohere = None  # type: ignore[assignment, misc]
    FakeListLLM = None  # type: ignore[assignment, misc]
    OpenAI = None  # type: ignore[assignment, misc]

__all__ = [
    "langchain",
    "BaseLLM",
    "FakeListLLM",
    "OpenAI",
    "AI21",
    "Cohere",
    "BaseChatModel",
    "ChatAnyscale",
    "ChatOpenAI",
    "ChatFireworks",
    "BaseLanguageModel",
    "Embeddings",
    "HuggingFaceEmbeddings",
    "HuggingFaceBgeEmbeddings",
    "PromptTemplate",
    "BasePromptTemplate",
    "ConditionalPromptSelector",
    "is_chat_model",
    "AIMessagePromptTemplate",
    "ChatPromptTemplate",
    "HumanMessagePromptTemplate",
    "BaseMessagePromptTemplate",
    "SystemMessagePromptTemplate",
    "StructuredTool",
    "Tool",
    "BaseTool",
    "BaseCallbackHandler",
    "BaseCallbackManager",
    "AIMessage",
    "FunctionMessage",
    "BaseMessage",
    "ChatMessage",
    "HumanMessage",
    "SystemMessage",
    "Document",
    "LLMResult",
]
