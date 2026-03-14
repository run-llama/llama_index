# LlamaIndex HADS Reader

Load [Human-AI Document Standard (HADS)](https://github.com/catcam/hads) files into LlamaIndex, filtering to only the tagged blocks your pipeline needs.

## Installation

```bash
pip install llama-index-readers-hads
```

## Usage

```python
from pathlib import Path
from llama_index.readers.hads import HADSReader

# Load only SPEC blocks (default) — ~70% token reduction
reader = HADSReader()
docs = reader.load_data(Path("architecture.hads.md"))

# Load SPEC + NOTE blocks
reader = HADSReader(block_types=["SPEC", "NOTE"])
docs = reader.load_data(Path("architecture.hads.md"))

# Load all block types
reader = HADSReader(block_types=["SPEC", "NOTE", "BUG", "?"])
docs = reader.load_data(Path("architecture.hads.md"))
```

## What is HADS?

HADS is a lightweight Markdown convention where documentation is tagged with semantic block types:

```markdown
**[SPEC]**
The cache uses LRU eviction with a 5-minute TTL.

**[NOTE]**
This was rewritten in Q3 to address memory growth.

**[BUG cache-invalidation]**
Cache is not invalidated on config reload.

**[?]**
Should the TTL be configurable per-key?
```

By loading only `SPEC` blocks (implementation facts), LLM context usage drops ~70% compared to loading the full document — without losing the information the model actually needs.

## Block Types

| Tag | Purpose |
|-----|---------|
| `SPEC` | Specification / implementation facts |
| `NOTE` | Background context, history, rationale |
| `BUG <title>` | Known bugs and workarounds |
| `?` | Open questions, unresolved decisions |

## Metadata

Each returned `Document` includes:

```python
{
    "source": "path/to/file.hads.md",
    "hads": True,
    "block_types": ["SPEC"],
    "blocks_found": 3,
    "block_tag": "SPEC",           # tag of this specific block
    "manifest": "Load SPEC for..." # from ## AI READING INSTRUCTION section
}
```
