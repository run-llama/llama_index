from rank_llm.rerank.reranker import Reranker

Reranker.create_model_coordinator(
    model_path="rank_zephyr", default_model_coordinator=None, interactive=False, vllm_batched=True
)
