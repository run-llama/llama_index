# LlamaIndex Postprocessor Integration: AWS Bedrock Rerankers

## Sample Usage

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor.bedrock_rerank import AWSBedrockRerank


documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
reranker = AWSBedrockRerank(
    top_n=3,
    model_id="cohere.rerank-v3-5:0",
    region_name="us-west-2",
)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

print(response)

print(response.source_nodes)
```
