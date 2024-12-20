from llama_index.extractors.marvin.base import MarvinMetadataExtractor


def test_marvin_class():
    """TODO: mock this class and have proper test."""
    assert MarvinMetadataExtractor.class_name() == "MarvinEntityExtractor"
