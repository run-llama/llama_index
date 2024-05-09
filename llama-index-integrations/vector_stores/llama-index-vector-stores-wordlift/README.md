<picture>
  <img alt="#Wordlift" width="200" src="https://eacn2n47zot.exactdn.com/wp-content/uploads/2022/12/logo.svg" style="margin-bottom: 25px;">
</picture>

# LlamaIndex Vector_Stores Integration: Wordlift

Wordlift is an AI-powered SEO platform. With our AI we build your own knowledge graph for your business with entities marked up by the different topics, categories and regions. Using this graph, search engines will be able to understand the structure of your content faster and more precisely.

This integration allows you to use Wordlift as a vector store for LlamaIndex to work with your Knowledge graph.

## Usage

In Wordlift Knowledge Graphs, the text nodes are stored using a specific embedding models, the nomic-embed-text-v1.\
Be sure to set nomic-embed-text-v1 as the embedding model, rather than relying on the default OpenAI embedding model.

Wordlift Knowledge Graphs are built on the principles of fully Linked Data, where each entity is assigned a permanent dereferentiable URI.\
When adding nodes to an existing Knowledge Graph, it's essential to include an "entity_id" in the metadata of each loaded document.\
For further insights into Fully Linked Data, explore these resources:
[W3C Linked Data](https://www.w3.org/DesignIssues/LinkedData.html),
[5 Star Data](https://5stardata.info/en/).

Please refer to the [notebook](../../../docs/docs/examples/vector_stores/wordlift_vector_store_demo.ipynb) for usage of Wordlift as vector store in LlamaIndex.
