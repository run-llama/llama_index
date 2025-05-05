import warnings
from llama_index.core import Document
from llama_index.core.llms import MockLLM
from llama_index.node_parser.slide import SlideNodeParser
from unittest.mock import patch

def test_empty_doc():
    """Ensure passing empty docs returns an empty List[TextNode]."""
    warnings.warn(
        "WARNING: This test may fail if the context length of MockLLM is changed.\n"
        "Make sure chunk_size * window_size fits within MockLLM.context_window.",
        UserWarning
    )

    llm = MockLLM()
    node_parser = SlideNodeParser.from_defaults(
        chunk_size=1300, # setting non default values to match context length of mock LLM
        window_size=3,
        llm=llm
    )
    nodes = node_parser.get_nodes_from_documents(documents=[Document(text="")])
    print(nodes)

    assert isinstance(nodes, list)
    assert nodes == []

def test_short_text_less_than_window():
    """Ensure parser handles short input without window overflow."""
    llm = MockLLM()

    # Window size is 5, but we only provide 2 sentences.
    parser = SlideNodeParser.from_defaults(
        llm=llm,
        chunk_size=780,
        window_size=5,
    )

    doc = Document(text="One. Two.")
    nodes = parser.get_nodes_from_documents([doc])

    assert isinstance(nodes, list)
    assert len(nodes) > 0  # Should still create valid nodes
    for node in nodes:
        assert "local_context" in node.metadata

def test_llm_called_expected_times():
    """Ensure LLM.chat() is called once per chunk (class‐level patching)."""
    # Prepare a document with 4 sentences
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    document = Document(text=text)

    mock_llm = MockLLM()

    # Patch the chat method on the class (so instance is bound to it cleanly)
    with patch.object(MockLLM, "chat", return_value="dummy context") as mock_chat:
        # Force each sentence to become its own chunk:
        parser = SlideNodeParser.from_defaults(
            llm=mock_llm,
            chunk_size=3,   # small enough that each sentence splits out
            window_size=1    # window of just the chunk itself
        )

        # Run parser
        nodes = parser.get_nodes_from_documents([document])

        # We expect one LLM.chat call per node returned
        assert mock_chat.call_count == len(nodes), (
            f"Expected chat() to be called {len(nodes)} times, "
            f"but got {mock_chat.call_count}"
        )
