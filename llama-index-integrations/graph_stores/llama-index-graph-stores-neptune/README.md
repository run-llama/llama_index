# LlamaIndex Graph_Stores Integration: Neptune

Amazon Neptune makes it easy to work with graph data in the AWS Cloud. Amazon Neptune includes both Neptune Database and Neptune Analytics.

Neptune Database is a serverless graph database designed for optimal scalability and availability. It provides a solution for graph database workloads that need to scale to 100,000 queries per second, Multi-AZ high availability, and multi-Region deployments. You can use Neptune Database for social networking, fraud alerting, and Customer 360 applications.

Neptune Analytics is an analytics database engine that can quickly analyze large amounts of graph data in memory to get insights and find trends. Neptune Analytics is a solution for quickly analyzing existing graph databases or graph datasets stored in a data lake. It uses popular graph analytic algorithms and low-latency analytic queries.

In this project, we integrate both Neptune Database and Neptune Analytics as the graph store to store the LlamaIndex graph data,

and use openCypher to query the graph data. so that people can use Neptune to interact with LlamaIndex graph index.

- Neptune Database

  - Property Graph Store: `NeptuneDatabasePropertyGraphStore`
  - Knowledge Graph Store: `NeptuneDatabaseGraphStore`

- Neptune Analytics
  - Property Graph Store: `NeptuneAnalyticsPropertyGraphStore`
  - Knowledge Graph Store: `NeptuneAnalyticsGraphStore`

## Installation

```shell
pip install llama-index llama-index-graph-stores-neptune
```

## Usage

### Property Graph Store

Please checkout this [tutorial](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_neptune.ipynb) to learn how to use Amazon Neptune with LlamaIndex.

### Knowledge Graph Store

Checkout this [tutorial](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/knowledge_graph/NeptuneDatabaseKGIndexDemo.ipynb) to learn how to use Amazon Neptune with LlamaIndex.
