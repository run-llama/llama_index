# LlamaIndex Packs Integration: LongRAG

This LlamaPack implements LongRAG based on [this paper](https://arxiv.org/pdf/2406.15319).

LongRAG retrieves large tokens at a time, with each retrieval unit being ~6k tokens long, consisting of entire documents or groups of documents. This contrasts the short retrieval units (100 word passages) of traditional RAG. LongRAG is advantageous because results can be achieved using only the top 4-8 retrieval units, and long-context LLMs can better understand the context of the documents because long retrieval units preserve their semantic integrity.

## Installation

```
# installation
pip install llama-index-packs-longrag

# source code
llamaindex-cli download-llamapack LongRAGPack --download-dir ./longrag_pack
```

## Code Usage

```py
from llama_index.packs.longrag import LongRAGPack
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

Settings.llm = OpenAI("gpt-4o")
embed_model = OpenAIEmbedding()

pack = LongRAGPack(data_dir="./data", embed_model=embed_model)

query_str = "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
res = pack.run(query_str)
print(str(res))
```
