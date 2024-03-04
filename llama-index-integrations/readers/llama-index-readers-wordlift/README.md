# WordLift Reader

```bash
pip install llama-index-readers-wordlift
```

The WordLift GraphQL Reader is a connector to fetch and transform data from a WordLift Knowledge Graph using your the WordLift Key. The connector provides a convenient way to load data from WordLift using a GraphQL query and transform it into a list of documents for further processing.

## Usage

To use the WordLift GraphQL Reader, follow the steps below:

1. Set up the necessary configuration options, such as the API endpoint, headers, query, fields, and configuration options (make sure you have with you the [Wordlift Key](https://docs.wordlift.io/pages/key-concepts/#wordlift-key)).
2. Create an instance of the `WordLiftLoader` class, passing in the configuration options.
3. Use the `load_data` method to fetch and transform the data.
4. Process the loaded documents as needed.

Here's an example of how to use the WordLift GraphQL Reader:

```python
import json
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from langchain.llms import OpenAI
from llama_index.readers.wordlift import WordLiftLoader

# Set up the necessary configuration options
endpoint = "https://api.wordlift.io/graphql"
headers = {
    "Authorization": "<YOUR_WORDLIFT_KEY>",
    "Content-Type": "application/json",
}

query = """
# Your GraphQL query here
"""
fields = "<YOUR_FIELDS>"
config_options = {
    "text_fields": ["<YOUR_TEXT_FIELDS>"],
    "metadata_fields": ["<YOUR_METADATA_FIELDS>"],
}
# Create an instance of the WordLiftLoader
reader = WordLiftLoader(endpoint, headers, query, fields, config_options)

# Load the data
documents = reader.load_data()

# Convert the documents
converted_doc = []
for doc in documents:
    converted_doc_id = json.dumps(doc.doc_id)
    converted_doc.append(
        Document(
            text=doc.text,
            doc_id=converted_doc_id,
            embedding=doc.embedding,
            doc_hash=doc.doc_hash,
            extra_info=doc.extra_info,
        )
    )

# Create the index and query engine
index = VectorStoreIndex.from_documents(converted_doc)
query_engine = index.as_query_engine()

# Perform a query
result = query_engine.query("<YOUR_QUERY>")

# Process the result as needed
logging.info("Result: %s", result)
```

This loader is designed to be used as a way to load data from WordLift KGs into [LlamaIndex](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/apify/actor#:~:text=load%20data%20into-,LlamaIndex,-and/or%20subsequently) and/or subsequently used as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
