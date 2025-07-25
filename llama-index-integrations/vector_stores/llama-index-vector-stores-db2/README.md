# LlamaIndex VectorStore Integration for IBM Db2

For a detailed guide look at https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/db2llamavs.ipynb

`pip install llama-index-vector-stores-db2`

# A sample example

```python
from typing import TYPE_CHECKING
import sys
from llama_index.core.schema import Document, TextNode
from llama_index.vector_stores.db2 import DB2LlamaVS, DistanceStrategy
from llama_index.vector_stores.db2 import base as db2llamavs

if TYPE_CHECKING:
    import ibm_db

"""
Create connection to Db2 database instance

The following sample code will show how to connect to Db2 Database. Besides the dependencies above, you will need a Db2 database instance (with version v12.1.2+, which has the vector datatype support) running.
"""

import ibm_db_dbi

database = ""
username = ""
password = ""

try:
    connection = ibm_db_dbi.connect(database, username, password)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!", e)


"""
Create Db2 vector store
"""

vectorstore = DB2LlamaVS.from_documents(
    client=conn,
    docs=chunks_with_mdata,
    table_name="db2vs",
    distance_strategy=DistanceStrategy.DOT_PRODUCT,
)

"""
Perform Similarity search
"""

# Similarity search
query = VectorStoreQuery(query_embedding=[1.0, 1.0], similarity_top_k=3)
results = vectorstore.query(query=query)
print(f"\n\n\nSimilarity search results for vector store: {results}\n\n\n")
```
