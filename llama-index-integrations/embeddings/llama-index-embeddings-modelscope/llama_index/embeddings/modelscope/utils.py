from typing import Dict, List
from llama_index.core.base.embeddings.base import Embedding


def sentence_to_input(sentence: str) -> Dict:
    return {
        "source_sentence": [sentence],
    }


def sentences_to_input(sentences: List[str]) -> Dict:
    return {
        "source_sentence": sentences,
    }


def output_to_embedding(output: Dict) -> Embedding:
    return output["text_embedding"][0].tolist()


def outputs_to_embeddings(outputs: Dict) -> List[Embedding]:
    return [output.tolist() for output in outputs["text_embedding"]]
