from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.vectara_query import VectaraQueryToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in VectaraQueryToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes

    tool_spec = VectaraQueryToolSpec(
        vectara_customer_id="1526022105",
        vectara_corpus_id="271",
        vectara_api_key="zqt_WvU_2c4b7yeZYT1PbDwy9Voh7xYTs_CANetTnA",
        citations_pattern="{doc.url}",
        summary_num_results=10,
    )

    # Debug with Ofer: will sometimes give citations (other times it will not)
    print(
        f'Tried query and got response: {tool_spec.rag_query("What can I learn in this course?")}\n\n'
    )

    print(
        f'Tried query and got response: {tool_spec.rag_query("Who is the lecturer for this course?")}\n\n'
    )

    print(
        f'Tried retrieval and got response: {tool_spec.semantic_search("What years does the information in this course cover?")}'
    )
