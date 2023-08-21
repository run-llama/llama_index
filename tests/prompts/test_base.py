"""Test prompts."""


from llama_index.bridge.langchain import ConditionalPromptSelector as LangchainSelector
from llama_index.bridge.langchain import PromptTemplate as LangchainTemplate
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM
from llama_index.prompts import LangchainPromptTemplate, PromptTemplate


def test_partial_format() -> None:
    """Test partial format."""
    prompt_txt = "hello {text} {foo}"
    prompt = PromptTemplate(prompt_txt)

    prompt_fmt = prompt.partial_format(foo="bar")

    assert isinstance(prompt_fmt, PromptTemplate)
    assert prompt_fmt.format(text="world") == "hello world bar"


def test_langchain_template() -> None:
    lc_template = LangchainTemplate.from_template("hello {text} {foo}")
    template = LangchainPromptTemplate(lc_template)

    template_fmt = template.partial_format(foo="bar")
    assert isinstance(template, LangchainPromptTemplate)
    assert template_fmt.format(text="world") == "hello world bar"


def test_langchain_selector_template() -> None:
    mock_llm = MockLLM()

    def is_mock(llm: LLM):
        return llm == mock_llm

    default_lc_template = LangchainTemplate.from_template("hello {text} {foo}")
    conditionals = [
        (is_mock, LangchainTemplate.from_template("hello {text} {foo} mock")),
    ]

    lc_selector = LangchainSelector(
        default_prompt=default_lc_template, conditionals=conditionals
    )
    template = LangchainPromptTemplate(selector=lc_selector)

    template_fmt = template.partial_format(foo="bar")
    assert isinstance(template, LangchainPromptTemplate)
    assert template_fmt.format(llm=mock_llm, text="world") == "hello world bar mock"
