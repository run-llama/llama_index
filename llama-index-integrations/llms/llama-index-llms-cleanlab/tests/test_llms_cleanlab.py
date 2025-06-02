from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.cleanlab import CleanlabTLM
from llama_index.llms.cleanlab.base import DEFAULT_MODEL, DEFAULT_MAX_TOKENS


def test_llms_cleanlab():
    names_of_base_classes = [b.__name__ for b in CleanlabTLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_init_defaults():
    llm = CleanlabTLM(api_key="x")

    assert llm.model == DEFAULT_MODEL
    assert llm.max_tokens == DEFAULT_MAX_TOKENS


def test_init_with_option_overrides():
    override_model = "gpt-4.1"
    override_max_tokens = 1024
    options = {"model": override_model, "max_tokens": override_max_tokens}
    llm = CleanlabTLM(api_key="x", options=options)

    assert llm.model == override_model
    assert llm.max_tokens == override_max_tokens
