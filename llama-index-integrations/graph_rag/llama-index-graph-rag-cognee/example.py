"""
Simple example demonstrating the CogneeGraphRAG integration.

This script shows how to:
1. Initialize the CogneeGraphRAG
2. Add documents to the knowledge graph
3. Process the data into a graph
4. Search for information
5. Visualize the graph

Requirements:
- Set OPENAI_API_KEY environment variable
- Install the package: pip install llama-index-graph-rag-cognee
"""

import asyncio
import os
from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


async def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🚀 Initializing CogneeGraphRAG...")

    # Initialize the GraphRAG system
    cognee_rag = CogneeGraphRAG(
        llm_api_key=api_key,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="kuzu",
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        relational_db_name="cognee_example_db",
    )

    print("📄 Creating sample documents...")

    # Create sample documents
    documents = [
        Document(
            text="Apple Inc. is a multinational technology company headquartered in Cupertino, California. "
            "It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
            "Apple is known for its consumer electronics, software, and online services."
        ),
        Document(
            text="Steve Jobs was the co-founder and longtime CEO of Apple Inc. "
            "He was known for his innovation in personal computing, animated movies, and mobile phones. "
            "Jobs passed away in 2011, leaving behind a legacy of revolutionary products."
        ),
        Document(
            text="The iPhone is Apple's flagship smartphone product, first released in 2007. "
            "It revolutionized the mobile phone industry with its touchscreen interface "
            "and App Store ecosystem. The iPhone runs on iOS operating system."
        ),
    ]

    print("➕ Adding documents to the knowledge graph...")

    # Add documents to the graph
    await cognee_rag.add(documents, dataset_name="apple_knowledge")
    print("   ✅ Documents added successfully")

    print("🔄 Processing data into knowledge graph...")

    # Process the data to create the knowledge graph
    await cognee_rag.process_data("apple_knowledge")
    print("   ✅ Data processed into graph")

    print("🔍 Searching the knowledge graph...")

    # Perform searches
    queries = [
        "Who founded Apple?",
        "When was iPhone released?",
        "What is Steve Jobs known for?",
    ]

    for query in queries:
        print(f"\n   Query: {query}")
        results = await cognee_rag.search(query)
        if results:
            print(f"   Answer: {results[0] if isinstance(results, list) else results}")
        else:
            print("   No results found")

    print("\n🕸️  Generating graph visualization...")

    # Create visualization (saves to home directory by default)
    try:
        viz_path = await cognee_rag.visualize_graph(
            open_browser=True, output_file_path="."
        )
        print(f"   ✅ Graph visualization saved to: {viz_path}")
        print(f"   🌐 Open the file in your browser to view the knowledge graph")
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")

    print("\n🎉 Example completed! The knowledge graph is ready for use.")
    print("\n📚 Next steps:")
    print("   - Add more documents with cognee_rag.add()")
    print("   - Process with cognee_rag.process_data()")
    print("   - Search with cognee_rag.search()")
    print("   - Explore related nodes with cognee_rag.get_related_nodes()")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
