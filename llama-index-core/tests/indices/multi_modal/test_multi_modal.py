import unittest

from llama_index.core import Document
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument
from unittest.mock import patch
import os


@unittest.skip("It seems there are no multimodel embeddings mock so yet. TBC")
@patch.dict(os.environ, {"IS_TESTING": "True"}, clear=True)
def test_async_multi_from_documents(mock_embed_model):
    documents = [Document(text=hex(i)[2:]) for i in range(16)] + [ImageDocument()]
    index = MultiModalVectorStoreIndex.from_documents(
        documents=documents,
        use_async=True,
        embed_model=mock_embed_model,
        image_embed_model="default",
        insert_batch_size=2,
        async_concurrency=2,
    )
    assert len(index.index_struct.nodes_dict) == 16
