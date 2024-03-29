# LlamaIndex Postprocessor Integration: Rankllm-Rerank

RankLLM offers a suite of listwise rerankers, albeit with focus on open source LLMs finetuned for the task. Currently, RankLLM supports 2 of these models: RankZephyr (`model="zephyr"`) and RankVicuna (`model="vicuna"`).

Please `pip install llama-index-postprocessor-rankllm-rerank` to install RankLLM rerank package.

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
    top_n=reranker_top_n, with_retrieval=with_retrieval,
    model=model
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
