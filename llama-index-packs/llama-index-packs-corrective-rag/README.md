# Corrective Retrieval Augmented Generation Llama Pack

This LlamaPack implements the Corrective Retrieval Augmented Generation (CRAG) [paper](https://arxiv.org/pdf/2401.15884.pdf)

Corrective Retrieval Augmented Generation (CRAG) is a method designed to enhance the robustness of language model generation by evaluating and augmenting the relevance of retrieved documents through a an evaluator and large-scale web searches, ensuring more accurate and reliable information is used in generation.

This LlamaPack uses [Tavily AI](https://app.tavily.com/home) API for web-searches. So, we recommend you to get the api-key before proceeding further.

### Installation

```bash
pip install llama-index llama-index-tools-tavily-research
```

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack CorrectiveRAGPack --download-dir ./corrective_rag_pack
```

You can then inspect the files at `./corrective_rag_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./corrective_rag_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
CorrectiveRAGPack = download_llama_pack(
    "CorrectiveRAGPack", "./corrective_rag_pack"
)

# You can use any llama-hub loader to get documents!
corrective_rag = CorrectiveRAGPack(documents, tavily_ai_api_key)
```

From here, you can use the pack, or inspect and modify the pack in `./corrective_rag_pack`.

The `run()` function contains around logic behind Corrective Retrieval Augmented Generation - [CRAG](https://arxiv.org/pdf/2401.15884.pdf) paper.

```python
response = corrective_rag.run("<query>", similarity_top_k=2)
```
