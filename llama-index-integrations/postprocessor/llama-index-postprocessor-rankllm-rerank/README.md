# LlamaIndex Postprocessor Integration: Rankllm-Rerank

RankLLM offers a suite of listwise rerankers, albeit with focus on open source LLMs finetuned for the task. Currently, RankLLM supports 2 of these models: RankZephyr (`model="zephyr"`) and RankVicuna (`model="vicuna"`). RankLLM also support RankGPT usage (`model="gpt"`, `gpt_model="VALID_OPENAI_MODEL_NAME"`).

Please `pip install llama-index-postprocessor-rankllm-rerank` to install RankLLM rerank package.

Parameters:

- top_n: Top N nodes to return from reranking.
- model: Reranker model name/class (`zephyr`, `vicuna`, or `gpt`).
- with_retrieval[Optional]: Perform retrieval before reranking with `Pyserini`.
- step_size[Optional]: Step size of sliding window for reranking large corpuses.
- gpt_model[Optional]: OpenAI model to use (e.g., `gpt-3.5-turbo`) if `model="gpt"`

### 💻 Example Usage

```
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-postprocessor-rankllm-rerank
```

First, build a vector store index with [llama-index](https://pypi.org/project/llama-index/).

```
index = VectorStoreIndex.from_documents(
    documents,
)
```

To set up the retriever and reranker:

```
query_bundle = QueryBundle(query_str)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=vector_top_k,
)

# configure reranker
reranker = RankLLMRerank(
    top_n=reranker_top_n,
    model=model,
    with_retrieval=with_retrieval,
    step_size=step_size,
    gpt_model=gpt_model,
)
```

To run retrieval+reranking:

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

### castorini/rank_llm

Repository for prompt-decoding using LLMs (`GPT3.5`, `GPT4`, `Vicuna`, and `Zephyr`)\
Website: [http://rankllm.ai](http://rankllm.ai)\
Stars: 193
