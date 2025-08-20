# LlamaIndex Node_Parser Integration: SlideNodeParser

Implements the SLIDE node parser described in the paper [SLIDE: Sliding Localized Information for Document Extraction](https://arxiv.org/abs/2503.17952), which introduces a chunking strategy that enriches segments with localized context from neighboring text. This improves downstream retrieval and question-answering tasks by preserving important contextual signals that might be lost with naive splitting.

`SlideNodeParser` implements a faithful adaptation of this technique using LLMs to generate a short context for each chunk based on its surrounding window.

Here's a summary of the method from the paper:

```
Traditional document chunking methods often truncate local context, weakening the semantic integrity of each chunk.
SLIDE introduces a sliding window approach that augments each chunk with a compact, LLM-generated summary of its surrounding context.

The process begins by greedily grouping sentences into base chunks based on a target token limit.
Then, for each chunk, a sliding window of neighboring chunks is selected (e.g., 5 before and after),
and the LLM is prompted to generate a brief context that situates the chunk within the overall document.

This context is then attached as metadata to each chunk, improving the quality of retrieval and generation tasks downstream, especially in Graph Retrieval Augmented generartion systems

Results from the research paper show that a single glean of the SlideNodeParser with default values (chunk_size=1200tokens, window_size=11) results in the identification of 37% more entities and relationships than standard Node Parsers
```

## Installation

```
pip install llama-index-node-parser-slide
```

## Usage

```python
from llama_index.core import Document
from llama_index.node_parser.slide import SlideNodeParser

# — Synchronous usage —
parser = SlideNodeParser.from_defaults(
    llm=llm,
    chunk_size=800,
    window_size=5,
)
nodes = parser.get_nodes_from_documents(
    [
        Document(text="document text 1"),
        Document(text="document text 2"),
    ]
)

# — Asynchronous usage (for parallel LLM calls) —
# Specify llm_workers > 1 to run multiple LLM calls concurrently
parser = SlideNodeParser.from_defaults(
    llm=llm,
    chunk_size=800,
    window_size=5,
    llm_workers=2,
)
nodes = await parser.aget_nodes_from_documents(
    [
        Document(text="document text 1"),
        Document(text="document text 2"),
    ]
)
```
