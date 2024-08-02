from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.vectara_query import VectaraQueryToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in VectaraQueryToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    tool_spec = VectaraQueryToolSpec(
        vectara_customer_id="3181727591",
        vectara_corpus_id="3",
        vectara_api_key="zqt_vaVPZ3bcwaDdq7l4LREAYegmEGSPGVe-TkfTSw",
    )

    print(
        f'Tried query and got response: {tool_spec.rag_query("What is the pet policy?")}\n\n'
    )

    print(
        f'Tried retrieval and got response: {tool_spec.semantic_search("What makes Velociraptors special?")}'
    )
