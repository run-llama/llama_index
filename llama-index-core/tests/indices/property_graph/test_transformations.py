import pytest
import logging
from unittest.mock import MagicMock
from llama_index.core.schema import TextNode
from llama_index.core.llms.mock import MockLLM
from llama_index.core.indices.property_graph.transformations.simple_llm import (
    SimpleLLMPathExtractor,
)
from llama_index.core.indices.property_graph.transformations.schema_llm import (
    SchemaLLMPathExtractor,
)
from llama_index.core.indices.property_graph.transformations.dynamic_llm import (
    DynamicLLMPathExtractor,
)


class FailingMockLLM(MockLLM):
    """Mock LLM that simulates an API error during prediction."""

    @property
    def metadata(self):
        return MagicMock()

    async def apredict(self, *args, **kwargs):
        raise ValueError("Simulated LLM apredict error")

    async def astructured_predict(self, *args, **kwargs):
        raise ValueError("Simulated LLM astructured_predict error")


@pytest.mark.asyncio
async def test_simple_llm_extractor_error_handling(caplog):
    node = TextNode(text="Test node")

    # Test graceful degradation (default)
    extractor = SimpleLLMPathExtractor(llm=FailingMockLLM(), raise_on_error=False)
    with caplog.at_level(logging.ERROR):
        result_nodes = await extractor.acall([node])

    assert len(result_nodes) == 1
    assert "Error during extraction: Simulated LLM apredict error" in caplog.text

    # Test strict raise
    extractor_strict = SimpleLLMPathExtractor(llm=FailingMockLLM(), raise_on_error=True)
    with pytest.raises(ValueError, match="Simulated LLM apredict error"):
        await extractor_strict.acall([node])


@pytest.mark.asyncio
async def test_schema_llm_extractor_error_handling(caplog):
    node = TextNode(text="Test node")

    # Test graceful degradation (default)
    extractor = SchemaLLMPathExtractor(llm=FailingMockLLM(), raise_on_error=False)
    with caplog.at_level(logging.ERROR):
        result_nodes = await extractor.acall([node])

    assert len(result_nodes) == 1
    assert (
        "Error during extraction: Simulated LLM astructured_predict error"
        in caplog.text
    )

    # Test strict raise
    extractor_strict = SchemaLLMPathExtractor(llm=FailingMockLLM(), raise_on_error=True)
    with pytest.raises(ValueError, match="Simulated LLM astructured_predict error"):
        await extractor_strict.acall([node])


@pytest.mark.asyncio
async def test_dynamic_llm_extractor_error_handling(caplog):
    node = TextNode(text="Test node")

    # Test graceful degradation (default)
    extractor = DynamicLLMPathExtractor(llm=FailingMockLLM(), raise_on_error=False)
    with caplog.at_level(logging.ERROR):
        result_nodes = await extractor.acall([node])

    assert len(result_nodes) == 1
    assert "Error during extraction: Simulated LLM apredict error" in caplog.text

    # Test strict raise
    extractor_strict = DynamicLLMPathExtractor(
        llm=FailingMockLLM(), raise_on_error=True
    )
    with pytest.raises(ValueError, match="Simulated LLM apredict error"):
        await extractor_strict.acall([node])


def test_schema_llm_extractor_prune_invalid_props_does_not_mutate_caller():
    extractor = SchemaLLMPathExtractor(llm=MockLLM())
    original_props = {"prop_a": "1", "prop_b": "2", "prop_c": "3"}
    props_copy = dict(original_props)

    pruned = extractor._prune_invalid_props(
        original_props, allowed_props=["prop_a", "prop_b"]
    )
    assert pruned == {"prop_a": "1", "prop_b": "2"}
    # Verify caller's dictionary is not modified
    assert original_props == props_copy
    assert "prop_c" in original_props

    # Verify None or empty allowed_props returns a copy rather than mutating or aliasing
    unrestricted = extractor._prune_invalid_props(original_props, allowed_props=None)
    assert unrestricted == original_props
    assert unrestricted is not original_props
