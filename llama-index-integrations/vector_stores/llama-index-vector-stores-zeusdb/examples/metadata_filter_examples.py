# metadata_filter_examples.py
from dotenv import load_dotenv

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.zeusdb import ZeusDBVectorStore

load_dotenv()

# Configure OpenAI
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4")

# Create ZeusDB vector store
vector_store = ZeusDBVectorStore(dim=1536, distance="cosine", index_type="hnsw")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create documents with rich metadata
documents = [
    Document(
        text="ZeusDB is a high-performance vector database optimized for "
        "semantic search.",
        metadata={
            "source": "doc",
            "topic": "zeusdb",
            "year": 2024,
            "tags": ["db", "vector", "search"],
        },
    ),
    Document(
        text="LlamaIndex provides RAG capabilities including retrievers, query "
        "engines, and rerankers.",
        metadata={
            "source": "doc",
            "topic": "llamaindex",
            "year": 2023,
            "tags": ["rag", "framework"],
        },
    ),
    Document(
        text="HNSW is a graph-based ANN index enabling fast approximate nearest "
        "neighbor search.",
        metadata={
            "source": "blog",
            "topic": "ann",
            "year": 2022,
            "tags": ["hnsw", "ann"],
        },
    ),
    Document(
        text="ZeusDB supports cosine distance and integrates with LlamaIndex as a "
        "vector store.",
        metadata={
            "source": "doc",
            "topic": "zeusdb",
            "year": 2025,
            "tags": ["integration", "vector"],
        },
    ),
    Document(
        text="BM25 and keyword methods focus on exact term matching rather than "
        "semantic similarity.",
        metadata={
            "source": "blog",
            "topic": "ir",
            "year": 2021,
            "tags": ["bm25", "keyword", "ir"],
        },
    ),
    Document(
        text="Vector search enables semantic similarity. It's commonly paired with "
        "metadata filters.",
        metadata={
            "source": "note",
            "topic": "search",
            "year": 2024,
            "tags": ["vector", "filters"],
        },
    ),
]

# Build index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("=== Metadata Filter Examples ===\n")

print("\n=== Example 1: No Filter (baseline) ===")
# First, test without filters to see if basic retrieval works
print("--- Filter: None ---\n")
retriever_baseline = index.as_retriever(similarity_top_k=5)
results_baseline = retriever_baseline.retrieve("ZeusDB database")

print(f"Found {len(results_baseline)} results without filters:")
for i, node in enumerate(results_baseline, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:100]}...")
    print(f"   Metadata: {node.metadata}\n")


print("\n=== Example 2: Topic and Year Filter ===")
# Test (a): topic == 'zeusdb' AND year >= 2024
print("--- Filter: topic=='zeusdb' AND year>=2024 ---\n")
filters_a = MetadataFilters.from_dicts(
    [
        {"key": "topic", "value": "zeusdb", "operator": FilterOperator.EQ},
        {"key": "year", "value": 2024, "operator": FilterOperator.GTE},
    ],
    condition=FilterCondition.AND,
)
retriever_a = index.as_retriever(similarity_top_k=5, filters=filters_a)
results_a = retriever_a.retrieve("integration with LlamaIndex")

for i, node in enumerate(results_a, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:100]}...")
    print(f"   Metadata: {node.metadata}\n")


print("\n=== Example 3: IN Operator - Multiple Values ===")
print("--- Filter: source IN ['blog', 'note'] ---\n")
filters_in = MetadataFilters.from_dicts(
    [
        {"key": "source", "value": ["blog", "note"], "operator": FilterOperator.IN},
    ],
    condition=FilterCondition.AND,
)
retriever_in = index.as_retriever(similarity_top_k=5, filters=filters_in)
results_in = retriever_in.retrieve("information retrieval methods")

print(f"Found {len(results_in)} documents from blogs or notes:")
for i, node in enumerate(results_in, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Source: {node.metadata.get('source')}")
    print(f"   Topic: {node.metadata.get('topic')}")
    print(f"   Text: {node.text[:80]}...\n")


print("\n=== Example 4: Array Contains Value ===")
print("--- Filter: tags CONTAINS 'vector' ---\n")
filters_contains = MetadataFilters.from_dicts(
    [
        {"key": "tags", "value": "vector", "operator": FilterOperator.CONTAINS},
    ],
    condition=FilterCondition.AND,
)
retriever_contains = index.as_retriever(similarity_top_k=5, filters=filters_contains)
results_contains = retriever_contains.retrieve("semantic search")

print(f"Found {len(results_contains)} documents tagged with 'vector':")
for i, node in enumerate(results_contains, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Tags: {node.metadata.get('tags')}")
    print(f"   Topic: {node.metadata.get('topic')}")
    print(f"   Text: {node.text[:80]}...\n")


print("\n=== Example 5: Exclusion Filter ===")
print("--- Filter: topic != 'zeusdb' AND year >= 2022 ---\n")
filters_ne = MetadataFilters.from_dicts(
    [
        {"key": "topic", "value": "zeusdb", "operator": FilterOperator.NE},
        {"key": "year", "value": 2022, "operator": FilterOperator.GTE},
    ],
    condition=FilterCondition.AND,
)
retriever_ne = index.as_retriever(similarity_top_k=5, filters=filters_ne)
results_ne = retriever_ne.retrieve("advanced search techniques")

print(f"Found {len(results_ne)} non-ZeusDB documents from 2022 onwards:")
for i, node in enumerate(results_ne, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Topic: {node.metadata.get('topic')} (not 'zeusdb')")
    print(f"   Year: {node.metadata.get('year')}")
    print(f"   Text: {node.text[:80]}...\n")


print("✓ Metadata filtering tests complete")
