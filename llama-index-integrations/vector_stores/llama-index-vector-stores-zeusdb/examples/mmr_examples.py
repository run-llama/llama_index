# examples/mmr_examples.py
"""
ZeusDB MMR (Maximal Marginal Relevance) Examples for LlamaIndex

Demonstrates using MMR to balance relevance and diversity in search results.

MMR is useful when you want to avoid redundant results and ensure diverse
perspectives in your retrieved documents.

Common use cases:
- RAG applications (avoiding repetitive context)
- Document summarization (covering different aspects)
- Research/exploration (finding varied perspectives)
- Question answering (providing comprehensive coverage)
"""

from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()

# Configure OpenAI
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

print("=" * 70)
print("ZeusDB MMR (Maximal Marginal Relevance) Examples")
print("=" * 70)
print()

# =============================================================================
# Example 1: The Problem - Redundant Results Without MMR
# =============================================================================
print("=" * 70)
print("Example 1: The Problem - Redundant Results")
print("=" * 70)
print()

# Create vector store
vector_store = ZeusDBVectorStore(dim=1536, distance="cosine")
embed_model = Settings.embed_model

# Create documents with some redundancy
documents = [
    # Cluster 1: Python programming (very similar)
    "Python is a high-level programming language known for its simplicity.",
    "Python is an easy-to-learn programming language with clear syntax.",
    "Python programming language is popular for its readability.",
    # Cluster 2: Python data science (similar to cluster 1)
    "Python is widely used in data science and machine learning.",
    "Data scientists prefer Python for analytics and ML tasks.",
    # Cluster 3: Other languages (different topic)
    "JavaScript is essential for web development and frontend applications.",
    "Java is a robust language used in enterprise applications.",
    "Rust provides memory safety without garbage collection.",
]

print(f"Adding {len(documents)} documents...")
nodes = []
for i, text in enumerate(documents):
    node = TextNode(text=text, metadata={"doc_id": i, "source": "examples"})
    node.embedding = embed_model.get_text_embedding(text)
    nodes.append(node)

vector_store.add(nodes)
print(f"  ✅ Added {len(nodes)} documents")
print()

# Query without MMR (standard similarity search)
query_text = "Python programming"
print(f"Query: '{query_text}'")
print()

query_embedding = embed_model.get_text_embedding(query_text)
query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)

print("📋 Standard Similarity Search (No MMR):")
results = vector_store.query(query_obj)

result_ids = results.ids or []
result_sims = results.similarities or []

# Create ID to node mapping
id_to_node = {node.node_id: node for node in nodes}

for i, (node_id, similarity) in enumerate(zip(result_ids, result_sims), 1):
    # Get the actual text using the mapping
    node = id_to_node.get(node_id)
    text = node.text if node else "Unknown"
    print(f"  {i}. Score: {similarity:.4f}")
    print(f"     {text[:70]}...")
    print()

print("⚠️  Notice: Top results are very similar (redundant information)")
print()

# =============================================================================
# Example 2: The Solution - Diverse Results With MMR
# =============================================================================
print("=" * 70)
print("Example 2: The Solution - MMR for Diversity")
print("=" * 70)
print()

print(f"Query: '{query_text}'")
print()

# Query WITH MMR
print("🎯 MMR Search (Balanced relevance + diversity):")
mmr_results = vector_store.query(
    query_obj,
    mmr=True,
    fetch_k=8,  # Fetch more candidates
    mmr_lambda=0.5,  # Balance: 0.5 relevance, 0.5 diversity
)

mmr_ids = mmr_results.ids or []
mmr_sims = mmr_results.similarities or []

for i, (node_id, similarity) in enumerate(zip(mmr_ids, mmr_sims), 1):
    # Get the actual text using the mapping
    node = id_to_node.get(node_id)
    text = node.text if node else "Unknown"
    print(f"  {i}. Score: {similarity:.4f}")
    print(f"     {text[:70]}...")
    print()

print("✅ Notice: Results are more diverse (different aspects/topics)")
print()

# =============================================================================
# Example 3: Tuning Lambda - Relevance vs Diversity Tradeoff
# =============================================================================
print("=" * 70)
print("Example 3: Tuning Lambda (Relevance vs Diversity)")
print("=" * 70)
print()

print("Lambda controls the relevance-diversity tradeoff:")
print("  • lambda=1.0: Pure relevance (like standard search)")
print("  • lambda=0.5: Balanced (default)")
print("  • lambda=0.0: Maximum diversity")
print()

lambda_values = [1.0, 0.7, 0.5, 0.3, 0.0]

for lambda_val in lambda_values:
    print(f"📊 MMR with lambda={lambda_val}:")

    mmr_results = vector_store.query(
        query_obj, mmr=True, fetch_k=8, mmr_lambda=lambda_val
    )

    mmr_ids = mmr_results.ids or []

    # Just show first 3 results
    print("  Top 3 results:")
    for i, node_id in enumerate(mmr_ids[:3], 1):
        node = id_to_node.get(node_id)
        text = node.text if node else "Unknown"
        print(f"    {i}. {text[:60]}...")
    print()

# =============================================================================
# Example 4: Practical RAG Scenario
# =============================================================================
print("=" * 70)
print("Example 4: Practical RAG Scenario")
print("=" * 70)
print()

# Create a new store with diverse tech articles
rag_store = ZeusDBVectorStore(dim=1536, distance="cosine")

articles = [
    # AI/ML cluster
    "Machine learning models require large datasets for training accuracy.",
    "Deep learning neural networks have revolutionized computer vision.",
    "AI ethics is becoming increasingly important in model development.",
    # Cloud/Infrastructure cluster
    "Cloud computing provides scalable infrastructure for applications.",
    "Kubernetes orchestrates containerized applications in production.",
    "Serverless computing reduces operational overhead significantly.",
    # Security cluster
    "Cybersecurity threats are evolving with sophisticated attack vectors.",
    "Zero-trust architecture is the modern approach to network security.",
    "Encryption protects sensitive data from unauthorized access.",
    # DevOps cluster
    "CI/CD pipelines automate software deployment and testing.",
    "Infrastructure as code enables reproducible deployments.",
    "Monitoring and observability are critical for system reliability.",
]

print(f"Creating knowledge base with {len(articles)} articles...")
rag_nodes = []
for i, text in enumerate(articles):
    node = TextNode(text=text, metadata={"doc_id": i, "source": "tech_articles"})
    node.embedding = embed_model.get_text_embedding(text)
    rag_nodes.append(node)

rag_store.add(rag_nodes)
print(f"  ✅ Added {len(rag_nodes)} articles")
print()

# Create ID to node mapping for RAG nodes
rag_id_to_node = {node.node_id: node for node in rag_nodes}

# User question
question = "How do modern software systems ensure quality and security?"
print(f"Question: '{question}'")
print()

query_embedding = embed_model.get_text_embedding(question)
query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=4)

# Standard search - might get redundant results
print("📋 Standard Search (Potential redundancy):")
standard_results = rag_store.query(query_obj)
standard_ids = standard_results.ids or []

for i, node_id in enumerate(standard_ids, 1):
    node = rag_id_to_node.get(node_id)
    text = node.text if node else "Unknown"
    print(f"  {i}. {text}")
print()

# MMR search - ensures diverse perspectives
print("🎯 MMR Search (Diverse perspectives for comprehensive answer):")
mmr_results = rag_store.query(
    query_obj,
    mmr=True,
    fetch_k=12,  # Larger candidate pool
    mmr_lambda=0.6,  # Slightly favor relevance
)
mmr_ids = mmr_results.ids or []

for i, node_id in enumerate(mmr_ids, 1):
    node = rag_id_to_node.get(node_id)
    text = node.text if node else "Unknown"
    print(f"  {i}. {text}")
print()

print("✅ MMR provides diverse context covering:")
print("   • Security aspects")
print("   • Quality/testing practices")
print("   • Infrastructure considerations")
print("   • Monitoring and reliability")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 70)
print("Summary: When to Use MMR")
print("=" * 70)
print()
print("✅ Use MMR when you want:")
print("  • Diverse results instead of similar/redundant ones")
print("  • Multiple perspectives on a topic")
print("  • Comprehensive coverage for RAG applications")
print("  • To avoid echo chamber effects in results")
print()
print("📊 Key Parameters:")
print("  • mmr=True: Enable MMR re-ranking")
print("  • fetch_k: Candidate pool size (default: 4*k, min: 20)")
print("  • mmr_lambda: Relevance/diversity tradeoff (default: 0.7)")
print("    - 1.0 = pure relevance (like standard search)")
print("    - 0.5 = balanced")
print("    - 0.0 = maximum diversity")
print()
print("💡 Best Practices:")
print("  • Set fetch_k >> k (e.g., fetch_k=20 for k=5)")
print("  • Start with lambda=0.7 (slightly favor relevance)")
print("  • Lower lambda for more diverse results")
print("  • Use with return_vector=True for better diversity calculation")
print()
print("⚠️  Trade-offs:")
print("  • MMR is slower than standard search (needs re-ranking)")
print("  • Requires fetch_k candidates (more initial results)")
print("  • May return less relevant results for diversity")
print()
print("📚 Common Use Cases:")
print("  • RAG systems: Diverse context for comprehensive answers")
print("  • Document exploration: Finding different perspectives")
print("  • Research: Covering multiple aspects of a topic")
print("  • Recommendation systems: Avoiding filter bubbles")
print()
