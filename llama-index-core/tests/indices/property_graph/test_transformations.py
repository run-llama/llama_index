import pytest
import logging
from unittest.mock import MagicMock
from llama_index.core.schema import TextNode
from llama_index.core.llms.mock import MockLLM
from llama_index.core.indices.property_graph.transformations.simple_llm import SimpleLLMPathExtractor
from llama_index.core.indices.property_graph.transformations.schema_llm import SchemaLLMPathExtractor
from llama_index.core.indices.property_graph.transformations.dynamic_llm import DynamicLLMPathExtractor

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
    assert "Error during extraction: Simulated LLM astructured_predict error" in caplog.text
    
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
    extractor_strict = DynamicLLMPathExtractor(llm=FailingMockLLM(), raise_on_error=True)
    with pytest.raises(ValueError, match="Simulated LLM apredict error"):
        await extractor_strict.acall([node])
