import os
from llama_index.core.download.dataset import download_llama_dataset

def test_llama_dataset() -> None:
    rag_dataset, documents = download_llama_dataset(
        "PaulGrahamEssayDataset", "./paul_graham"
    )
    assert documents == 1
