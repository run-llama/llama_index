import platformdirs  # noqa: F401
from datetime import datetime

from llama_index.core.prompts.rich import RichPromptTemplate
from llama_index.core.schema import TextNode


def test_basic_rich_prompt():
    prompt = RichPromptTemplate("Hello, {{name}}!")

    assert not prompt.is_chat_template
    assert prompt.template_vars == ["name"]

    formatted_prompt = prompt.format(name="John")
    assert formatted_prompt == "Hello, John!"

    formatted_prompt = prompt.format(name="Jane")
    assert formatted_prompt == "Hello, Jane!"


def test_basic_rich_chat_prompt():
    prompt = RichPromptTemplate("{% chat role='user' %}Hello, {{name}}!{% endchat %}")

    assert prompt.is_chat_template

    messages = prompt.format_messages(name="John")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello, John!"


def test_function_mapping():
    def today(**prompt_args):
        return datetime.now().strftime("%Y-%m-%d")

    prompt = RichPromptTemplate(
        "Hello, {{name}}, today is {{today}}", function_mappings={"today": today}
    )

    messages = prompt.format_messages(name="John")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello, John, today is " + datetime.now().strftime(
        "%Y-%m-%d"
    )


def test_object_mapping():
    nodes = [
        TextNode(text="You are a helpful assistant."),
        TextNode(text="You are new to the company."),
        TextNode(text="You are a great assistant."),
    ]
    prompt_str = """
Hello, {{name}}. Here is some information about you:

{% for node in nodes %}
- {{node.text}}
{% endfor %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(name="John", nodes=nodes)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert nodes[0].text in messages[0].content
    assert nodes[1].text in messages[0].content
    assert nodes[2].text in messages[0].content


def test_prompt_with_images():
    image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

    prompt_str = """
{% chat role='user' %}
Hello, {{name}}. Here is an image of you:

{{ your_image | image }}

{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(name="John", your_image=image_url)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 2
    assert messages[0].blocks[0].block_type == "text"
    assert messages[0].blocks[1].block_type == "image"
    assert str(messages[0].blocks[1].url) == image_url
