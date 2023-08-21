"""Test prompts."""


from llama_index.bridge.langchain import PromptTemplate as LangchainTemplate
from llama_index.llms.base import LLM
from llama_index.llms.openai import OpenAI
from llama_index.prompts import LangchainPromptTemplate, PromptTemplate


def is_openai(llm: LLM) -> bool:
    """Test condition."""
    return isinstance(llm, OpenAI)


def test_partial_format() -> None:
    """Test partial format."""
    prompt_txt = "hello {text} {foo}"
    prompt = PromptTemplate(prompt_txt)

    prompt_fmt = prompt.partial_format(foo="bar")

    assert isinstance(prompt_fmt, PromptTemplate)
    assert prompt_fmt.format(text="world") == "hello world bar"


def test_langchain_prompt() -> None:
    lc_template = LangchainTemplate.from_template("hello {text} {foo}")
    template = LangchainPromptTemplate(lc_template)

    template_fmt = template.partial_format(foo="bar")
    assert isinstance(template, LangchainPromptTemplate)
    assert template_fmt.format(text="world") == "hello world bar"
