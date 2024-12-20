# RAFT: Adapting Language Model to Domain Specific RAG Llama Pack

This LlamaPack implements RAFT: Adapting Language Model to Domain Specific RAG [paper](https://arxiv.org/abs/2403.10131)

Retrieval Augmented FineTuning (RAFT) is a training recipe introduced in this paper that aims to improve the performance of large language models (LLMs) in open-book, in-domain question-answering tasks. Given a question and a set of retrieved documents, RAFT trains the LLM to identify and cite verbatim the most relevant sequences from the documents that help answer the question, while ignoring irrelevant or distracting information. By explicitly training the model to distinguish between relevant and irrelevant information and to provide evidence from the relevant documents, RAFT encourages the LLM to develop better reasoning and explanation abilities, ultimately improving its ability to answer questions accurately and rationally in scenarios where additional context or knowledge is available.

A key component of RAFT is how the dataset is generated for fine-tuning. Each QA pair also includes an "oracle" document from which the answer to the question can be deduced as well as "distractor" documents which are irrelevant. During training this forces the model to learn which information is relevant/irrelevant and also memorize domain knowledge.

We've implemented the dataset generation part in a LlamaPack. Check out our [full notebook here](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb).

### Installation

```bash
pip install llama-index
```

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack RAFTDatasetPack --download-dir ./raft_dataset_pack
```

You can then inspect the files at `./raft_dataset_pack` and use them as a template for your own project.

## Code Usage

You can download the pack to a the `./raft_dataset_pack` directory:

```python
from llama_index.core.llama_pack import download_llama_pack

# download and install dependencies
RAFTDatasetPack = download_llama_pack("RAFTDatasetPack", "./raft_dataset_pack")

# You can use any llama-hub loader to get documents!
raft_dataset = RAFTDatasetPack(file_path)
```

From here, you can use the pack, or inspect and modify the pack in `./raft_dataset_pack`.

The `run()` function contains around logic behind RAFT: Adapting Language Model to Domain Specific RAG [paper](https://arxiv.org/abs/2403.10131)

```python
dataset = raft_dataset.run()
```

This will return the dataset which can be further used for finetuned purpose. Please refer to [original blog](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674) on using the dataset for fine-tuning.
