from typing import Dict, Type

from llama_index.text_splitter.code_splitter import CodeSplitter
from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from llama_index.text_splitter.types import TextSplitter

RECOGNIZED_TEXT_SPLITTERS: Dict[str, Type[TextSplitter]] = {
    SentenceSplitter.class_name(): SentenceSplitter,
    TokenTextSplitter.class_name(): TokenTextSplitter,
    CodeSplitter.class_name(): CodeSplitter,
}


def load_text_splitter(data: dict) -> TextSplitter:
    text_splitter_name = data.get("class_name", None)
    if text_splitter_name is None:
        raise ValueError("TextSplitter loading requires a class_name")

    if text_splitter_name not in RECOGNIZED_TEXT_SPLITTERS:
        raise ValueError(f"Invalid TextSplitter name: {text_splitter_name}")

    return RECOGNIZED_TEXT_SPLITTERS[text_splitter_name].from_dict(data)
