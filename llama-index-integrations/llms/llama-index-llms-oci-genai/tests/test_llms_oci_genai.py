from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.oci_genai import OCIGenAI


def test_oci_genai_embedding_class():
    names_of_base_classes = [b.__name__ for b in OCIGenAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
