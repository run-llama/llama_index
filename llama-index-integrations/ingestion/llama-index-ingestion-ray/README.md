# LlamaIndex Ingestion: Ray

**A Scalable LlamaIndex ingestion pipeline powered by Ray.**

This integration uses Ray’s distributed compute framework to parallelize document transformations (parsing, chunking, and embedding), enabling high-throughput processing for large-scale datasets.

## Installation

```bash
pip install llama-index-integrations-ray
```

## Usage

Distribute the workload across your Ray cluster by wrapping transformations in `RayTransformComponent` objects and passing them to `RayIngestionPipeline`.

```python
import ray
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.ingestion.ray import (
    RayIngestionPipeline,
    RayTransformComponent,
)

# Start a new cluster (or connect to an existing one, see https://docs.ray.io/en/latest/ray-core/configure.html)
ray.init()

# Create transformations
transformations = [
    RayTransformComponent(SentenceSplitter, chunk_size=25, chunk_overlap=0),
    RayTransformComponent(
        transform_class=TitleExtractor,
        map_batches_kwargs={
            "batch_size": 10,  # Define the batch size
            # "num_cpus": 4  # The number of CPUs to reserve for each parallel map worker.
            # "num_gpus": 1  # The number of GPUs to reserve for each parallel map worker.
            # See https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html for all the available parameters
        },
    ),
    RayTransformComponent(
        transform_class=OpenAIEmbedding,
        map_batches_kwargs={
            "batch_size": 10,
        },
    ),
]

# Create the Ray ingestion pipeline
pipeline = RayIngestionPipeline(transformations=transformations)

# Run the pipeline with many documents
nodes = pipeline.run(documents=[Document.example()] * 10)
```
