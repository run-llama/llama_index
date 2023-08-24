from llama_index.text_splitter.code_splitter import CodeSplitter
from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.text_splitter.token_splitter import TokenTextSplitter
from llama_index.text_splitter.types import TextSplitter


def load_text_splitter(data: dict) -> TextSplitter:
    text_splitter_name = data.get("class_name", None)
    if text_splitter_name is None:
        raise ValueError("TextSplitter loading requires a class_name")

    if text_splitter_name == SentenceSplitter.__name__:
        return SentenceSplitter.from_dict(data)
    elif text_splitter_name == TokenTextSplitter.__name__:
        return TokenTextSplitter.from_dict(data)
    elif text_splitter_name == CodeSplitter.__name__:
        return CodeSplitter.from_dict(data)
    else:
        raise ValueError(f"Unknown text splitter name: {text_splitter_name}")
