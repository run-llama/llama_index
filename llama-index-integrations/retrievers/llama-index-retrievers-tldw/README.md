## TL;DW Video Retriever

### Overview

**TL;DW** is a powerful video understanding API that retrieves precise moments from videos using natural language queries. By integrating **TL;DW** with **LlamaIndex**, we can efficiently index and search video content, enabling seamless knowledge retrieval from videos.

### Setup

- Obtain API keys from [tl;dw Playground](https://app.trytldw.ai/account?tab=api). New users are granted free indexing minutes automatically.

- Install the required packages:

```sh
pip install llama-index-retrievers-tldw
```

### Usage

- Initialize the TldwRetriever with your API key and collection ID:

```python
from llama_index.retrievers.tldw import TldwRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Initialize the retriever
retriever = TldwRetriever(
    api_key="YOUT_TLDW_API_KEY",
    collection_id="YOUR_COLLECTION_ID",  # Replace with your actual collection ID
)

# Create a query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
)

# Query and summarize response
response = query_engine.query("What are the brands of smart watches reviewed?")
print(
    response
)  # "The brands of smartwatches reviewed in the videos are Apple and Garmin."
```

## Support

If you have any questions or feedback, please feel free to reach out to us.

- [tl;dw AI](https://www.trytldw.ai/)
- [Code Examples](https://github.com/tldw-ai/example-playbooks)
- [Email](mailto:contact@trytldw.ai)
