from rank_llm.rerank.reranker import Reranker

Reranker.create_agent(
    model_path="rank_zephyr", default_agent=None, interactive=False, vllm_batched=True
)
