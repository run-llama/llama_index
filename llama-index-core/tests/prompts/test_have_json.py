from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import (
    PromptTemplate,
)
from llama_index.core.prompts.utils import get_template_vars


def test_template_hava_json() -> None:
    """Test partial format."""
    prompt_txt = 'hello {text} {foo} \noutput format:\n```json\n{"name": "llamaindex"}\n```'
    except_prompt = 'hello world bar \noutput format:\n```json\n{"name": "llamaindex"}\n```'

    prompt_template = PromptTemplate(prompt_txt)
    template_vars = get_template_vars(prompt_txt)
    prompt_fmt = prompt_template.partial_format(foo="bar")
    prompt = prompt_fmt.format(text="world")

    assert isinstance(prompt_fmt, PromptTemplate)
    assert template_vars == ["text", "foo"]
    assert prompt == except_prompt
    assert prompt_fmt.format_messages(text="world") == [
        ChatMessage(content=except_prompt, role=MessageRole.USER)
    ]
