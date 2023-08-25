from typing import List, Optional

from llama_index.llms.base import LLM
from llama_index.node_parser.extractors.metadata_extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    EntityExtractor,
    TitleExtractor,
    KeywordExtractor,
)


def load_extractor(
    data: dict,
    extractors: Optional[List[MetadataExtractor]] = None,
    llm: Optional[LLM] = None,
) -> MetadataExtractor:
    extractor_name = data.get("class_name", None)
    if extractor_name is None:
        raise ValueError("Extractor loading requires a class_name")

    # remove unused key
    data.pop("is_text_node_only", None)

    if extractor_name == MetadataExtractor.__name__:
        return MetadataExtractor.from_dict(data, extractors=extractors)
    elif extractor_name == SummaryExtractor.__name__:
        return SummaryExtractor.from_dict(data, llm=llm)
    elif extractor_name == QuestionsAnsweredExtractor.__name__:
        return QuestionsAnsweredExtractor.from_dict(data, llm=llm)
    elif extractor_name == EntityExtractor.__name__:
        return EntityExtractor.from_dict(data)
    elif extractor_name == TitleExtractor.__name__:
        return TitleExtractor.from_dict(data, llm=llm)
    elif extractor_name == KeywordExtractor.__name__:
        return KeywordExtractor.from_dict(data, llm=llm)
    else:
        raise ValueError(f"Unknown extractor name: {extractor_name}")
