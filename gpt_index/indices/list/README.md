## 🔗 GPTListIndex

### Index Construction

GPTListIndex is a simple list-based data structure. During index construction, GPTListIndex takes in a dataset of text documents as input, chunks them up into smaller document chunks, and concatenates them into a list. GPT is not called at all during index construction.

### Query

During query-time, GPT List Index constructs an answer using the _create and refine_ paradigm. An initial answer to the query is constructed using the first text chunk. The answer is then _refined_ through feeding in subsequent text chunks as context. Refinement could mean keeping the original answer, making small edits to the original answer, or rewriting the original answer completely.

**Usage**

```python
from gpt_index import GPTListIndex, SimpleDirectoryReader

# build index
documents = SimpleDirectoryReader('data').load_data()
index = GPTListIndex(documents)
# save index
index.save_to_disk('index_list.json')
# load index from disk
index = GPTListIndex.load_from_disk('index_list.json')
# query
response = index.query("<question text>")

```
