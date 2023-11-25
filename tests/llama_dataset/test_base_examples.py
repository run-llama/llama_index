"""Test llama data examples."""

from llama_index.llama_dataset.base import (
    LlamaRagDataExample,
    LlamaRagDataExampleKind,
)


def test_repr_rag_example_class() -> None:
    query = "This is a test query, is it not?"
    response = "Yes it is."
    contexts = ["This is a sample context"]
    rag_example = LlamaRagDataExample(
        query=query,
        response=response,
        contexts=contexts,
        kind=LlamaRagDataExampleKind.HUMAN,
    )

    expected_repr = (
        "LlamaRagDataExample(query='This is a test query, is it not?', "
        "response='Yes it is.', contexts=['This is a sample context'], "
        "kind=<LlamaRagDataExampleKind.HUMAN: 'human'>, reference=None)"
    )
    assert repr(rag_example) == expected_repr
    assert rag_example.class_name == "LlamaRagDataExample"
