from llama_index.core.extractors.interface import BaseExtractor
from llama_index.extractors.entity import EntityExtractor


def test_extractor_class():
    ext = EntityExtractor()
    assert isinstance(ext, BaseExtractor)
