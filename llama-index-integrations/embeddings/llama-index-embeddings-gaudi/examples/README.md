# GaudiEmbedding Examples

This folder contains examples showcasing how to use LlamaIndex with Embeddings integration `llama_index.embeddings.gaudi.GaudiEmbedding` on Intel Gaudi.

## Installation

### On Intel Gaudi

```bash
pip install --upgrade-strategy eager optimum[habana]
pip install llama-index-embeddings-gaudi
```

## List of Examples

### Basic Usage Example

The example [basic.py](./basic.py) shows how to run `GaudiEmbedding` on Intel Gaudi and conduct embedding tasks such as text and query embedding. Run the example as following:

```bash
PT_HPU_LAZY_ACC_PAR_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true python basic.py
```

### Graph RAG Example

GrapgRAG combines Graph Analysis and Retrieval Augmented Generation for richly understanding text datasets. With `GaudiHuggingFaceEmbeddings` and `GaudiLLM`, you may now run GraphRAG using local LLM on Intel Gaudi. The example [graphrag.py](./graphrag.py) shows how to create a knowledge graph from unstructured text and use that graph to retrieve relevant information for generative tasks via graph search. Follow the instructions to run the example:

#### Starting NEO4J Database Server

```
docker run --restart always --publish=7474:7474 --publish=7687:7687 --env NEO4J_AUTH=neo4j/<neo4j-server-password> -v $PWD/data:/data -v $PWD/plugins:/plugins --name neo4j-apoc -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4JLABS_PLUGINS=\[\"apoc\"\] -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* neo4j:5.22.0
```

#### Additional Dependencies

```bash
# Intel Gaudi Software Version 1.18.0 or later is required.
pip install llama-index-llms-huggingface
pip install llama-index-llms-gaudi
pip install requirements.txt

# Set the following environment variables:
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=<neo4j-server-password> #default: neo4j
export NEO4J_URL=neo4j://<neo4j-server-host-ip>:7687
export NEO4J_DATABASE=neo4j

PT_HPU_LAZY_ACC_PAR_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true python graphrag.py
```
