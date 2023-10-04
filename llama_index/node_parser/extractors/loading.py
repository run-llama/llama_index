from typing import List, Optional

from llama_index.llms.base import LLM
from llama_index.node_parser.extractors.metadata_extractors import (
    EntityExtractor,
    KeywordExtractor,
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)


def load_extractor(
    data: dict,
    extractors: Optional[List[MetadataExtractor]] = None,
    llm: Optional[LLM] = None,
) -> MetadataExtractor:
    extractor_name = data.get("class_name", None)
    if extractor_name is None:
        raise ValueError("Extractor loading requires a class_name")

    if extractor_name == MetadataExtractor.class_name():
        return MetadataExtractor.from_dict(data, extractors=extractors)
    elif extractor_name == SummaryExtractor.class_name():
        return SummaryExtractor.from_dict(data, llm=llm)
    elif extractor_name == QuestionsAnsweredExtractor.class_name():
        return QuestionsAnsweredExtractor.from_dict(data, llm=llm)
    elif extractor_name == EntityExtractor.class_name():
        return EntityExtractor.from_dict(data)
    elif extractor_name == TitleExtractor.class_name():
        return TitleExtractor.from_dict(data, llm=llm)
    elif extractor_name == KeywordExtractor.class_name():
        return KeywordExtractor.from_dict(data, llm=llm)
    else:
        raise ValueError(f"Unknown extractor name: {extractor_name}")
