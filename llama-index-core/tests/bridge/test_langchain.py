import importlib
import sys
from types import ModuleType


def _install_module(monkeypatch, name: str, **attributes) -> None:
    module = ModuleType(name)
    module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_missing_optional_langchain_models_are_none(monkeypatch) -> None:
    placeholder = object()
    _install_module(monkeypatch, "langchain")
    _install_module(monkeypatch, "langchain_core")
    _install_module(
        monkeypatch, "langchain_core.language_models", BaseLanguageModel=placeholder
    )
    _install_module(monkeypatch, "langchain_core.callbacks")
    _install_module(
        monkeypatch,
        "langchain_core.callbacks.base",
        BaseCallbackHandler=placeholder,
        BaseCallbackManager=placeholder,
    )
    _install_module(monkeypatch, "langchain_classic")
    _install_module(monkeypatch, "langchain_classic.chains")
    _install_module(
        monkeypatch,
        "langchain_classic.chains.prompt_selector",
        ConditionalPromptSelector=placeholder,
        is_chat_model=placeholder,
    )
    _install_module(monkeypatch, "langchain.chat_models")
    _install_module(
        monkeypatch, "langchain.chat_models.base", BaseChatModel=placeholder
    )
    _install_module(monkeypatch, "langchain_core.documents")
    _install_module(monkeypatch, "langchain_core.documents.base", Document=placeholder)
    _install_module(monkeypatch, "langchain_core.outputs", LLMResult=placeholder)
    _install_module(
        monkeypatch,
        "langchain_core.prompts",
        PromptTemplate=placeholder,
        BasePromptTemplate=placeholder,
    )
    _install_module(
        monkeypatch,
        "langchain_core.prompts.chat",
        AIMessagePromptTemplate=placeholder,
        BaseMessagePromptTemplate=placeholder,
        ChatPromptTemplate=placeholder,
        HumanMessagePromptTemplate=placeholder,
        SystemMessagePromptTemplate=placeholder,
    )
    _install_module(
        monkeypatch,
        "langchain_core.messages",
        AIMessage=placeholder,
        BaseMessage=placeholder,
        ChatMessage=placeholder,
        FunctionMessage=placeholder,
        HumanMessage=placeholder,
        SystemMessage=placeholder,
    )
    _install_module(monkeypatch, "langchain_core.embeddings", Embeddings=placeholder)
    _install_module(
        monkeypatch,
        "langchain_core.tools",
        BaseTool=placeholder,
        StructuredTool=placeholder,
        Tool=placeholder,
    )
    _install_module(monkeypatch, "langchain_community")
    _install_module(monkeypatch, "langchain_community.chat_models")
    module_name = "llama_index.core.bridge.langchain"
    previous_module = sys.modules.pop(module_name, None)
    try:
        bridge = importlib.import_module(module_name)

        assert bridge.ChatOpenAI is None
        assert bridge.ChatFireworks is None
        assert bridge.ChatAnyscale is None
        assert bridge.HuggingFaceBgeEmbeddings is None
        assert bridge.HuggingFaceEmbeddings is None
        assert bridge.AI21 is None
        assert bridge.BaseLLM is None
        assert bridge.Cohere is None
        assert bridge.FakeListLLM is None
        assert bridge.OpenAI is None
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
