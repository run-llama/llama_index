# Automated Metadata Extraction for Nodes

You can use LLMs to automate metadata extraction with our `MetadataExtractor` modules.

Our metadata extractor modules include the following "feature extractors":
- `SummaryExtractor` - automatically extracts a summary over a set of Nodes
- `QuestionsAnsweredExtractor` - extracts a set of questions that each Node can answer
- `TitleExtractor` - extracts a title over the context of each Node
- `EntityExtractor` - extracts entities (i.e. names of places, people, things) mentioned in the content of each Node

You can use these feature extractors within our overall `MetadataExtractor` class. Then you can plug in the `MetadataExtractor` into our node parser:

```python
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    TitleExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
    ],
)

node_parser = SimpleNodeParser.from_defaults(
    text_splitter=text_splitter,
    metadata_extractor=metadata_extractor,
)
# assume documents are defined -> extract nodes
nodes = node_parser.get_nodes_from_documents(documents)
```


```{toctree}
---
caption: Metadata Extraction Guides
maxdepth: 1
---
/examples/metadata_extraction/MetadataExtractionSEC.ipynb
/examples/metadata_extraction/EntityExtractionClimate.ipynb
```