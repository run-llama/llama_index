# Metadata Extraction


## Introduction
In many cases, especially with long documents, a chunk of text may lack the context necessary to disambiguate the chunk from other similar chunks of text. 

To combat this, we use LLMs to extract certain contextual information relevant to the document to better help the retrieval and language models disambiguate similar-looking passages.

We show this in an [example notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/metadata_extraction/MetadataExtractionSEC.ipynb) and demonstrate its effectiveness in processing long documents.

## Usage

First, we define a metadata extractor that takes in a list of feature extractors that will be processed in sequence.

We then feed this to the node parser, which will add the additional metadata to each node.
```python
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
    ],
)

node_parser = SimpleNodeParser(
    text_splitter=text_splitter,
    metadata_extractor=metadata_extractor,
)
```

Here is an example of the enriched medadata:

```
```

## Custom Extractors

If the provided extractors do not fit your needs, you can also define a custom extractor like so:
```python
from llama_index.node_parser.extractors import MetadataFeatureExtractor

class CustomExtractor(MetadataFeatureExtractor):
    def extract(self, nodes) -> List[Dict]:
        metadata_list = [
            {
                "custom": node.metadata["document_title"]
                + "\n"
                + node.metadata["excerpt_keywords"]
            }
            for node in nodes
        ]
        return metadata_list
```

In a more advanced example, it can also make use of an `llm_predictor` to extract features from the node content and the existing metadata. Refer to the [source code of the provided metadata extractors](https://github.com/jerryjliu/llama_index/blob/main/llama_index/node_parser/extractors/metadata_extractors.py) for more details.