# Semantic Scholar Loader

```bash
pip install llama-index-readers-semanticscholar

pip install llama-index-llms-openai
```

Welcome to Semantic Scholar Loader. This module serves as a crucial utility for researchers and professionals looking to get scholarly articles and publications from the Semantic Scholar database.

For any research topic you are interested in, this loader reads relevant papers from a search result in Semantic Scholar into `Documents`.

Please go through [demo_s2.ipynb](demo_s2.ipynb)

## Some preliminaries -

- `query_space` : broad area of research
- `query_string` : a specific question to the documents in the query space

**UPDATE** :

To download the open access pdfs and extract text from them, simply mark the `full_text` flag as `True` :

```python
s2reader = SemanticScholarReader()
documents = s2reader.load_data(query_space, total_papers, full_text=True)
```

## Usage

Here is an example of how to use this loader in `llama_index` and get citations for a given query.

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.readers.semanticscholar import SemanticScholarReader

s2reader = SemanticScholarReader()

# narrow down the search space
query_space = "large language models"

# increase limit to get more documents
documents = s2reader.load_data(query=query_space, limit=10)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
)
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

# query the index
response = query_engine.query("limitations of using large language models")
print("Answer: ", response)
print("Source nodes: ")
for node in response.source_nodes:
    print(node.node.metadata)
```

### Output

```bash
Answer:  The limitations of using large language models include the struggle to learn long-tail knowledge [2], the need for scaling by many orders of magnitude to reach competitive performance on questions with little support in the pre-training data [2], and the difficulty in synthesizing complex programs from natural language descriptions [3].
Source nodes:
{'venue': 'arXiv.org', 'year': 2022, 'paperId': '3eed4de25636ac90f39f6e1ef70e3507ed61a2a6', 'citationCount': 35, 'openAccessPdf': None, 'authors': ['M. Shanahan'], 'title': 'Talking About Large Language Models'}
{'venue': 'arXiv.org', 'year': 2022, 'paperId': '6491980820d9c255b9d798874c8fce696750e0d9', 'citationCount': 31, 'openAccessPdf': None, 'authors': ['Nikhil Kandpal', 'H. Deng', 'Adam Roberts', 'Eric Wallace', 'Colin Raffel'], 'title': 'Large Language Models Struggle to Learn Long-Tail Knowledge'}
{'venue': 'arXiv.org', 'year': 2021, 'paperId': 'a38e0f993e4805ba8a9beae4c275c91ffcec01df', 'citationCount': 305, 'openAccessPdf': None, 'authors': ['Jacob Austin', 'Augustus Odena', 'Maxwell Nye', 'Maarten Bosma', 'H. Michalewski', 'David Dohan', 'Ellen Jiang', 'Carrie J. Cai', 'Michael Terry', 'Quoc V. Le', 'Charles Sutton'], 'title': 'Program Synthesis with Large Language Models'}
```
