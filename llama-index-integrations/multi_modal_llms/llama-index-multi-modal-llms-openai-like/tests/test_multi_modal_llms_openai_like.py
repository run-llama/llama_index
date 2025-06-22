from llama_index.llms.openai_like import OpenAILike
from llama_index.multi_modal_llms.openai_like import OpenAILikeMultiModal


def test_multi_modal_class():
    names_of_base_classes = [b.__name__ for b in OpenAILikeMultiModal.__mro__]
    assert OpenAILike.__name__ in names_of_base_classes
