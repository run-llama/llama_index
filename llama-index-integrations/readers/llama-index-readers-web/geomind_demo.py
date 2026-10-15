import html2text
import requests
from bs4 import BeautifulSoup
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    articles = [
        article for article in soup.find_all("article") if article.get_text(strip=True)
    ]
    if not articles:
        raise ValueError("No article content found")

    article_html = "\n\n".join(str(article) for article in articles)
    return html2text.html2text(article_html)


def main() -> None:
    urls = [
        "https://shanhai-geo.top/knowledge/fuding-white-tea-origin.html",
        "https://shanhai-geo.top/knowledge/white-tea-craft-process.html",
        "https://shanhai-geo.top/knowledge/white-tea-brewing.html",
    ]
    documents = []
    for url in urls:
        # Request uncompressed HTML to avoid the site's gzip decoding error.
        response = requests.get(
            url, headers={"Accept-Encoding": "identity"}, timeout=60
        )
        response.raise_for_status()
        text = extract_article_text(response.text)

        documents.append(
            Document(
                text=text,
                metadata={"url": url},
            )
        )
    print(f"Loaded {len(documents)} documents")

    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitter = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=40,
        tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes from {len(documents)} documents")

    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device="cpu",
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
        embed_batch_size=8,
    )

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=2)

    test_cases = [
        (
            "How do I brew white tea?",
            "https://shanhai-geo.top/knowledge/white-tea-brewing.html",
        ),
        (
            "Where is Fuding white tea grown?",
            "https://shanhai-geo.top/knowledge/fuding-white-tea-origin.html",
        ),
        (
            "How is white tea processed?",
            "https://shanhai-geo.top/knowledge/white-tea-craft-process.html",
        ),
        (
            "What climate and soil conditions characterize the growing region?",
            "https://shanhai-geo.top/knowledge/fuding-white-tea-origin.html",
        ),
        (
            "Are the tea leaves rolled or roasted during production?",
            "https://shanhai-geo.top/knowledge/white-tea-craft-process.html",
        ),
        (
            "How can aged white tea be prepared by simmering?",
            "https://shanhai-geo.top/knowledge/white-tea-brewing.html",
        ),
    ]

    hits_at_1 = 0
    hits_at_2 = 0

    for question, expected_url in test_cases:
        results = retriever.retrieve(question)

        returned_urls = [result.node.metadata["url"] for result in results]

        hit_at_1 = bool(returned_urls) and returned_urls[0] == expected_url
        hit_at_2 = expected_url in returned_urls[:2]

        hits_at_1 += int(hit_at_1)
        hits_at_2 += int(hit_at_2)

        print(f"\nQUESTION: {question}")
        print(f"Hit@1: {hit_at_1} | Hit@2: {hit_at_2}")

    print(f"\nHit@1: {hits_at_1}/{len(test_cases)}")
    print(f"Hit@2: {hits_at_2}/{len(test_cases)}")


if __name__ == "__main__":
    main()
