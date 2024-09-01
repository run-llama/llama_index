<picture>
  <img alt="#WordLift" width="200" src="https://wordlift.io/wp-content/uploads/2022/12/logo.svg" style="margin-bottom: 25px;">
</picture>

# LlamaIndex Vector_Stores Integration: WordLift

WordLift is an AI-powered SEO platform. With our AI we build your own knowledge graph for your business with entities marked up by the different topics, categories and regions. Using this graph, search engines will be able to understand the structure of your content faster and more precisely. To access a WordLift key and unlock our AI-powered SEO tools, visit [WordLift](https://wordlift.io/).

This integration enables the use of WordLift as a vector store for LlamaIndex, allowing you to work with your knowledge graph directly from your codebase.

## Features

- Perform retrieval-augmented generation (RAG) using your knowledge graph data directly in your codebase.
- Add new nodes and search within your knowledge graph effortlessly.

## Usage

Please refer to the [notebook](./examples/wordlift_vector_store_demo.ipynb) for usage of WordLift as vector store in LlamaIndex.

WordLift Knowledge Graphs are built on the principles of fully Linked Data, where each entity is assigned a permanent dereferentiable URI.\
When adding nodes to an existing Knowledge Graph, it's essential to include an "entity_id" in the metadata of each loaded document.\
For further insights into Fully Linked Data, explore these resources:
[W3C Linked Data](https://www.w3.org/DesignIssues/LinkedData.html),
[5 Star Data](https://5stardata.info/en/).
