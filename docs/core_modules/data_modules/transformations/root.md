# Transformations for Nodes + Documents

Transformations are features that modify documents and nodes, whether by splitting text and creating multiple nodes, adding metadata, and more!

Every transformation has one reqruiement: the input and output is always `List[BaseNode]`. This means that every transformation is chainable with another.

In LlamaIndex, there are several classes of transformations:

- **NodeParsers** -- These transformations take a node or document, and split them based on content or length. Think of it as a one-to-many mapping.
- **MetadataExtractors** -- Metadata extractors take a node or document, and use their content as input to function that extract metadata about that content (summaries, questions/answers, keywords, etc.). This metadata can then be used to augment embeddings and LLM interactions. These transformations output the same number of inputs, but with added metadata.
- **Embeddings** -- Embeddings take the content of the node, and generate a vector representation of that node. The embedding is attached to the node, and can be used in vector indexes for retrieval.
- **Custom** -- Since the API for a transformation is extremely simple, you can extend the base class to create your own transformations.

## Usage Pattern

Using transformations directly is easy!

```python
from llama_index.ingesstion import run_transformations
from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SentenceSplitter

documents = ...

transformations = [
    SentenceSplitter(chunk_size=256),
    OpenAIEmbedding(embed_batch_size=10)
]

nodes = run_transformations(
    documents,
    transformations,
    in_place=True,
    show_progress=True,
)
```

See more detailed usage patterns below.

```{toctree}
---
maxdepth: 1
---
node_parser_usage_pattern.md
metadata_extractor_usage_pattern.md
/core_modules/model_modules/embeddings/usage_pattern.md
custom_usage_pattern.md
```

## Modules

See detailed guides for each transformation here:

```{toctree}
---
maxdepth: 1
---
metadata_extractor_modules.md
node_parser_modules.md
/core_modules/model_modules/embeddings/modules.md
```
