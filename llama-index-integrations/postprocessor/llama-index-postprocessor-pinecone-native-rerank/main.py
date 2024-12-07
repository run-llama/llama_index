import os

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.pinecone_native_rerank import PineconeNativeRerank


os.environ["PINECONE_API_KEY"] = "xxx"

m = PineconeNativeRerank(top_n=10)

assert m.model == "bge-reranker-v2-m3"
assert m.top_n == 10
