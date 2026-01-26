# LlamaIndex Readers Integration: Causal

Reader for `.causal` binary knowledge graph files with embedded deterministic inference.

## Overview

The `.causal` format solves the fundamental problem of AI-assisted discovery: **LLMs hallucinate, databases don't reason**.

| Technology | What it does | What's missing |
|------------|--------------|----------------|
| **SQLite** | Stores facts | No reasoning |
| **Vector RAG** | Finds similar text | No logic |
| **LLMs** | Reasons creatively | Hallucination risk |
| **.causal** | Stores + Reasons | **Zero hallucination** |

### Key Features

- **30-40x faster queries** than SQLite (pre-computed inference)
- **50-200% fact amplification** through transitive chains
- **Zero hallucination** - pure deterministic logic with full provenance
- **Edge AI ready** - compact enough for mobile/offline use

## Installation

```bash
pip install llama-index-readers-causal
```

## Usage

```python
from llama_index.readers.causal import CausalReader

# Initialize reader
reader = CausalReader(
    include_inferred=True,  # Include derived facts
    min_confidence=0.5,     # Filter by confidence
)

# Load all triplets from a .causal file
documents = reader.load_data("knowledge.causal")

# Or search for specific topics
documents = reader.load_data(
    "knowledge.causal",
    query="COVID fatigue",
    limit=10,
)

# Each document contains a triplet
for doc in documents:
    print(doc.text)
    # [INFERRED] SARS-CoV-2 → damages → mitochondria → causes → fatigue

    print(doc.metadata)
    # {'trigger': 'SARS-CoV-2', 'mechanism': 'causes', 'outcome': 'fatigue',
    #  'confidence': 0.85, 'is_inferred': True, 'provenance': [...]}
```

### With Query Engine

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.causal import CausalReader

# Load knowledge graph
reader = CausalReader()
documents = reader.load_data("knowledge.causal")

# Build index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Query with zero-hallucination grounding
response = query_engine.query(
    "What mechanisms connect COVID to chronic fatigue?"
)
```

## Document Metadata

Each document includes rich metadata:

| Field | Type | Description |
|-------|------|-------------|
| `trigger` | str | The cause/trigger entity |
| `mechanism` | str | The relationship type |
| `outcome` | str | The effect/outcome entity |
| `confidence` | float | Confidence score (0-1) |
| `is_inferred` | bool | Whether derived or explicit |
| `provenance` | list | Source triplets for inferred facts |
| `source` | str | Original source (e.g., paper) |

## References

- **PyPI**: https://pypi.org/project/dotcausal/
- **GitHub**: https://github.com/DT-Foss/dotcausal
- **Whitepaper**: https://doi.org/10.5281/zenodo.18326222
