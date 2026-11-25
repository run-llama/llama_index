import pytest

from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("cutoff", [-1.0, 0.5])
async def test_similarity_postprocessor_drops_nan(
    use_async: bool, cutoff: float
) -> None:
    invalid = NodeWithScore(node=TextNode(text="invalid"), score=float("nan"))
    valid = NodeWithScore(node=TextNode(text="valid"), score=0.8)
    processor = SimilarityPostprocessor(similarity_cutoff=cutoff)

    if use_async:
        results = await processor.apostprocess_nodes([invalid, valid])
    else:
        results = processor.postprocess_nodes([invalid, valid])

    assert len(results) == 1
    assert results[0] is valid


def test_similarity_postprocessor_preserves_cutoff_boundary() -> None:
    scores = [None, -float("inf"), 0.49, 0.5, 0.8, float("inf")]
    nodes = [
        NodeWithScore(node=TextNode(text=str(i)), score=score)
        for i, score in enumerate(scores)
    ]

    results = SimilarityPostprocessor(similarity_cutoff=0.5).postprocess_nodes(nodes)

    assert results == nodes[3:]
