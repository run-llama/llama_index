# LlamaIndex Postprocessor Integration: Google Rerankers

Integrating llama-index with [Google rerankers](https://cloud.google.com/generative-ai-app-builder/docs/ranking#genappbuilder_rank-python).

It is essential to have a project in Google Cloud Platform (GCP) and loads its credentials as explained in [here](https://google-auth.readthedocs.io/en/master/reference/google.oauth2.service_account.html).

## Sample Usage

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor.google_rerank import GoogleRerank


documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
reranker = GoogleRerank(
    top_n=3,
    model_id="semantic-ranker-512-003",
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
