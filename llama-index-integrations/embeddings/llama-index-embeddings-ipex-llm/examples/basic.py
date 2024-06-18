import argparse
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IpexLLMEmbedding Basic Usage Example")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="The huggingface repo id for the embedding model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "xpu"],
        help="The device (Intel CPU or Intel GPU) the embedding model runs on",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency.",
        help="The sentence you prefer for text embedding",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is IPEX-LLM?",
        help="The sentence you prefer for query embedding",
    )

    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    text = args.text
    query = args.query

    # load the embedding model on Intel GPU with IPEX-LLM optimizations
    embedding_model = IpexLLMEmbedding(model_name=model_name, device=device)

    text_embedding = embedding_model.get_text_embedding(text)
    print(f"embedding[:10]: {text_embedding[:10]}")

    text_embeddings = embedding_model.get_text_embedding_batch([text, query])
    print(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
    print(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

    query_embedding = embedding_model.get_query_embedding(query)
    print(f"query_embedding[:10]: {query_embedding[:10]}")
