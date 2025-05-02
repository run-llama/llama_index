from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenAIMultiModal.__mro__]
    assert OpenAI.__name__ in names_of_base_classes
