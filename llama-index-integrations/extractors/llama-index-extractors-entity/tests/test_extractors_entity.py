from llama_index.extractors.entity import EntityExtractor


def test_extractor_class():
    # TODO: mock the entity extractor
    assert EntityExtractor.class_name() == "EntityExtractor"
