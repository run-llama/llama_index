# LlamaIndex Graph RAG Integration: Cognee

Cognee assists developers in introducing greater predictability and management into their Retrieval-Augmented Generation (RAG) workflows through the use of graph architectures, vector stores, and auto-optimizing pipelines. Displaying information as a graph is the clearest way to grasp the content of your documents. Crucially, graphs allow systematic navigation and extraction of data from documents based on their hierarchy.

This integration provides a seamless interface between LlamaIndex and Cognee, enabling you to:

- **Build Knowledge Graphs** from your documents automatically
- **Search with Multiple Methods** including vector search, graph traversal, and hybrid approaches
- **Visualize Your Data** with interactive HTML graph visualizations
- **Scale with Enterprise Databases** including PostgreSQL, Neo4j, and more

For more information, visit [Cognee documentation](https://docs.cognee.ai/)

## Installation

```shell
pip install llama-index-graph-rag-cognee
```

## Usage

### Basic Example

```python
import os
import asyncio
from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


async def main():
    # Initialize CogneeGraphRAG
    cognee_rag = CogneeGraphRAG(
        llm_api_key=os.environ["OPENAI_API_KEY"],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="kuzu",  # or "neo4j", "networkx"
        vector_db_provider="lancedb",
        relational_db_provider="sqlite",
        relational_db_name="cognee_db",
    )

    # Create documents
    documents = [
        Document(
            text="Apple Inc. is a technology company founded by Steve Jobs."
        ),
        Document(
            text="Steve Jobs was the CEO of Apple and known for innovation."
        ),
        Document(text="The iPhone was released by Apple in 2007."),
    ]

    # Add documents to the knowledge graph
    await cognee_rag.add(documents, dataset_name="apple_knowledge")

    # Process data into the knowledge graph
    await cognee_rag.process_data("apple_knowledge")

    # Search the knowledge graph
    results = await cognee_rag.search("Who founded Apple?")
    print("Search Results:", results)

    # Generate and visualize the knowledge graph
    viz_path = await cognee_rag.visualize_graph(
        open_browser=True,  # Automatically open in browser
        output_file_path=".",  # Save to current directory
    )
    print(f"Visualization saved to: {viz_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

```python
import os
import pandas as pd
import asyncio
from llama_index.core import Document
from llama_index.graph_rag.cognee import CogneeGraphRAG


async def advanced_example():
    # Load news data
    news = pd.read_csv(
        "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    )[:5]

    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for i, row in news.iterrows()
    ]

    # Initialize with enterprise databases
    cognee_rag = CogneeGraphRAG(
        llm_api_key=os.environ["OPENAI_API_KEY"],
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        graph_db_provider="neo4j",  # Enterprise graph database
        vector_db_provider="qdrant",  # Scalable vector database
        relational_db_provider="postgresql",
        relational_db_name="cognee_production",
    )

    # Build knowledge graph
    await cognee_rag.add(documents, "news_dataset")
    await cognee_rag.process_data("news_dataset")

    # Multiple search approaches
    print("=== Graph-based Search ===")
    graph_results = await cognee_rag.search(
        "Tell me about the people mentioned"
    )
    for result in graph_results:
        print(f"📊 {result}")

    print("\n=== RAG-based Search ===")
    rag_results = await cognee_rag.rag_search(
        "Tell me about the people mentioned"
    )
    for result in rag_results:
        print(f"🔍 {result}")

    print("\n=== Related Nodes ===")
    related = await cognee_rag.get_related_nodes("person")
    for node in related:
        print(f"🔗 {node}")

    # Generate visualization
    await cognee_rag.visualize_graph(
        open_browser=False, output_file_path="/path/to/your/output/directory"
    )


if __name__ == "__main__":
    asyncio.run(advanced_example())
```

## Key Features

### 🔍 Multiple Search Methods

- **`search()`** - Graph-based search using knowledge graph relationships
- **`rag_search()`** - Traditional RAG search using vector similarity
- **`get_related_nodes()`** - Find connected entities and relationships

### 📊 Graph Visualization

- **Interactive HTML visualization** of your knowledge graph
- **Automatic browser opening** for immediate viewing
- **Customizable output paths** for saving visualizations
- **Built on D3.js** for rich, interactive exploration

### 🏗️ Flexible Architecture

- **Multiple database backends** for different scale requirements
- **Async-first design** for high-performance applications
- **LlamaIndex Document integration** for seamless workflows
- **Enterprise-ready** with PostgreSQL, Neo4j, and Qdrant support

### 🎯 Dataset Management

- **Organize data by datasets** for logical separation
- **Process datasets independently** for better control
- **Future support** for advanced dataset operations

## API Reference

### CogneeGraphRAG Methods

| Method                | Description                          | Parameters                                                 |
| --------------------- | ------------------------------------ | ---------------------------------------------------------- |
| `add()`               | Add documents to the knowledge graph | `data`: Documents, `dataset_name`: String                  |
| `process_data()`      | Process data into graph structure    | `dataset_names`: String                                    |
| `search()`            | Graph-based search                   | `query`: String                                            |
| `rag_search()`        | Vector similarity search             | `query`: String                                            |
| `get_related_nodes()` | Find related entities                | `node_id`: String                                          |
| `visualize_graph()`   | Generate HTML visualization          | `open_browser`: Bool, `output_file_path`: Optional[String] |

## Supported Databases

**Relational databases:** SQLite, PostgreSQL

**Vector databases:** LanceDB, PGVector, QDrant, Weaviate

**Graph databases:** Neo4j, NetworkX, Kuzu
