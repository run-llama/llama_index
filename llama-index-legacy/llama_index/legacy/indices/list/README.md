## 🔗 SummaryIndex

### Index Construction

SummaryIndex is a simple list-based data structure. During index construction, SummaryIndex takes in a dataset of text documents as input, chunks them up into smaller document chunks, and concatenates them into a list. GPT is not called at all during index construction.

### Query

During query-time, Summary Index constructs an answer using the _create and refine_ paradigm. An initial answer to the query is constructed using the first text chunk. The answer is then _refined_ through feeding in subsequent text chunks as context. Refinement could mean keeping the original answer, making small edits to the original answer, or rewriting the original answer completely.

**Usage**

```python
from llama_index.legacy import SummaryIndex, SimpleDirectoryReader

# build index
documents = SimpleDirectoryReader("data").load_data()
index = SummaryIndex.from_documents(documents)
# query
query_engine = index.as_query_engine()
response = query_engine.query("<question text>")
```
