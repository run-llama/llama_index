# AIMon Rerank

AIMon Rerank is a postprocessor for [LlamaIndex](https://github.com/run-llama/llama_index) that leverages the AIMon API to rerank retrieved documents based on contextual relevance. It refines document retrieval by applying a custom task definition and returning the most contextually relevant nodes.

## Features

- **Domain Adaptable Reranking:** Applies a user-defined task to assess document relevance.
- **Batch Processing:** Efficiently handles text in batches to stay within word count limit of 10000 per batch.
- **Seamless Integration:** Easily integrates with LlamaIndex query engine.

## Installation

Ensure you have Python 3.8+ installed. Then, install the required packages:

```bash
pip install llama-index
pip install llama-index-postprocessor-aimon-rerank
```

## Setup

Set your AIMon API key as an environment variable (or pass it directly when instantiating the reranker):

```bash
export AIMON_API_KEY="your_aimon_api_key_here"
```

## Basic Usage

Below is a minimal example demonstrating how to use AIMon Rerank with LlamaIndex:

```python
import os
from llama_index.postprocessor.aimon_rerank import AIMonRerank
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents from a directory.
documents = SimpleDirectoryReader(
    "data/your_documents/example_of_afforestion"
).load_data()

# Build a vector store index from the documents.
index = VectorStoreIndex.from_documents(documents=documents)

# Define a task for the reranker.
task_definition = "Determine the relevance of context documents with respect to the user query."

# Initialize AIMonRerank, with the following parameters:
# top_n: After reranking, the top_n most contextually relevant nodes are selected for response generation.
# api_key: Ensure the AIMON_API_KEY is set, either directly or as an  environment variable.
# task_definition: The task definition serves as an explicit instruction that defines what the reranking evaluation should focus on.
aimon_rerank = AIMonRerank(
    top_n=2,
    api_key=os.environ["AIMON_API_KEY"],
    task_definition=task_definition,
)

# Create a query engine with the AIMon reranking postprocessor.
# For example, the query engine retrieves top 10 most relevant nodes, out of which only top_n are selected after reranking.
query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[aimon_rerank]
)

# Execute a query.
response = query_engine.query("What did the protagonist do in this essay?")
pprint_response(response, show_source=True)
```

## Output

### Final Response

The protagonist was responsible for planting 1000 trees.

---

#### Source Node 1/2

**Node ID:** 2940ea4a-69ec-4fc4-9dd4-8ed54a9d4f1b
**Similarity:** 0.49260445005911023

**Text:**
The protagonist took on the responsibility of afforestation in their village, initiating a large-scale tree-planting campaign. Over several months, they coordinated volunteers, secured funding, and ensured the successful planting of 1000 trees in barren lands to restore the local ecosystem.

---

#### Source Node 2/2

**Node ID:** 0baaf5af-6e6b-4889-8407-e49d1753980c
**Similarity:** 0.45151918284717965

**Text:**
Determined to combat deforestation, the protagonist spearheaded a green initiative, setting an ambitious goal of planting 1000 trees. Through meticulous planning and relentless effort, they managed to achieve their objective, significantly improving the area's biodiversity and air quality.
