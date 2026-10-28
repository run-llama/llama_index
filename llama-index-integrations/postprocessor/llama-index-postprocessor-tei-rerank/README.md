# LlamaIndex Postprocessor Integration: TEI Rerank

Re-Rankers hosted on Text Embedding Inference Serve by Huggingface.

Install TEI Rerank package with:
`pip install llama-index-postprocessor-tei-rerank`

_text-embeddings-inference_ v0.4.0 added support for CamemBERT, RoBERTa and XLM-RoBERTa Sequence Classification models. Please refer to their repo for any further clarrification :
https://github.com/huggingface/text-embeddings-inference

## Docker start-up for TEI:

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --auto-truncate
```

After the TEI server starts, use an existing `retriever` and a query string
`prompt` to retrieve and rerank nodes as follows. The Docker `--model-id` option
selects the model served by TEI.

```python
from llama_index.core import QueryBundle
from llama_index.postprocessor.tei_rerank import TextEmbeddingInference as TEIR

query_bundle = QueryBundle(prompt)
retrieved_nodes = retriever.retrieve(query_bundle)

postprocessor = TEIR(top_n=5, base_url="http://127.0.0.1:8080")

reranked_nodes = postprocessor.postprocess_nodes(
    nodes=retrieved_nodes, query_bundle=query_bundle
)
```
