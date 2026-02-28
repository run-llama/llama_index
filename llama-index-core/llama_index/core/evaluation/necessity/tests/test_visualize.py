from llama_index.core.evaluation.necessity.visualize import (
    NecessityTextRenderer,
)
from llama_index.core.evaluation.necessity.graph import (
    EvidenceNecessityGraph,
)
from llama_index.core.evaluation.necessity.necessity import (
    NecessityResult,
)


def test_text_renderer_output():
    graph = EvidenceNecessityGraph()

    graph.add_result(
        NecessityResult(
            claim="The Eiffel Tower is in Paris.",
            initially_supported=True,
            necessary_context_indices=[0],
        )
    )

    contexts = [
        "The Eiffel Tower is located in Paris.",
        "The Eiffel Tower is a landmark.",
    ]

    renderer = NecessityTextRenderer()
    output = renderer.render(
        graph=graph,
        contexts=contexts,
    )

    assert "CLAIM: The Eiffel Tower is in Paris." in output
    assert "Fragile" in output
    assert "Context [0]" in output
