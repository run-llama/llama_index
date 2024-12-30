# LlamaIndex Postprocessor Integration: Rankllm-Rerank

RankLLM offers a suite of rerankers, albeit with focus on open source LLMs finetuned for the task. To use a model offered by the RankLLM suite, pass the desired model's **Hugging Face model path**, found at [Castorini's Hugging Face](https://huggingface.co/castorini).

e.g., to access `LiT5-Distill-base`, pass [`castorini/LiT5-Distill-base`](https://huggingface.co/castorini/LiT5-Distill-base) as the model name.

For more information about RankLLM and the models supported, visit **[rankllm.ai](http://rankllm.ai)**. Please `pip install llama-index-postprocessor-rankllm-rerank` to install RankLLM rerank package.

#### Parameters:

- `model`: Reranker model name
- `top_n`: Top N nodes to return from reranking
- `window_size`: Reranking window size. Applicable only for listwise and pairwise models.
- `batch_size`: Reranking batch size. Applicable only for pointwise models.

#### Model Coverage

Below are all the rerankers supported with the model name to be passed as an argument to the constructor. Some model have convenience names for ease of use:

**Listwise**:

- **RankZephyr**. model=`rank_zephyr` or `castorini/rank_zephyr_7b_v1_full`
- **RankVicuna**. model=`rank_zephyr` or `castorini/rank_vicuna_7b_v1`
- **RankGPT**. Takes in a _valid_ gpt model. e.g., `gpt-3.5-turbo`, `gpt-4`,`gpt-3`
- **LiT5 Distill**. model=`castorini/LiT5-Distill-base`
- **LiT5 Score**. model=`castorini/LiT5-Score-base`

**Pointwise**:

- MonoT5. model='monot5'

### 💻 Example Usage

```
pip install llama-index-core
pip install llama-index-llms-openai
from llama_index.postprocessor.rankllm_rerank import RankLLMRerank
```

First, build a vector store index with [llama-index](https://pypi.org/project/llama-index/).

```
index = VectorStoreIndex.from_documents(
    documents,
)
```

To set up the _retriever_ and _reranker_:

```
query_bundle = QueryBundle(query_str)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=vector_top_k,
)

# configure reranker
reranker = RankLLMRerank(
    model=model_name
    top_n=reranker_top_n,
)
```

To run _retrieval+reranking_:

```
# retrieve nodes
retrieved_nodes = retriever.retrieve(query_bundle)

# rerank nodes
reranked_nodes = reranker.postprocess_nodes(
    retrieved_nodes, query_bundle
)
```

### 🔧 Dependencies

Currently, RankLLM rerankers require `CUDA` and for `rank-llm` to be installed (`pip install rank-llm`). The built-in retriever, which uses [Pyserini](https://github.com/castorini/pyserini), requires `JDK11`, `PyTorch`, and `Faiss`.

### `castorini/rank_llm`

Repository for prompt-decoding using LLMs: **[http://rankllm.ai](http://rankllm.ai)**
