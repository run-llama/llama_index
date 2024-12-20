"""Contributor Retriever #1.

A retriever over some synthetic 'symptom2disease' examples.
"""

import os
from llama_index.core import VectorStoreIndex
from llama_index.core.llama_dataset.simple import LabelledSimpleDataset
from llama_index.core.schema import TextNode


# load the synthetic dataset
synthetic_dataset = LabelledSimpleDataset.from_json(
    "./data/contributor1_synthetic_dataset.json"
)


nodes = [
    TextNode(text=el.text, metadata={"reference_label": el.reference_label})
    for el in synthetic_dataset[:]
]

index = VectorStoreIndex(nodes=nodes)
similarity_top_k = int(os.environ.get("SIMILARITY_TOP_K"))
retriever = index.as_retriever(similarity_top_k=similarity_top_k)
