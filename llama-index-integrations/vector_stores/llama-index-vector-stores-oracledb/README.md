# LlamaIndex VectorStore Integration for Oracle

This is a very basic example on how to use Oracle as a vector store with llamaindex. For a detailed guide look at https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/orallamavs.ipynb

`pip install llama-index-vector-stores-oracledb`

# A sample example

```python
from typing import TYPE_CHECKING
import sys
from llama_index.core.schema import Document, TextNode
from llama_index.readers.oracleai import OracleReader, OracleTextSplitter
from llama_index.embeddings.oracleai import OracleEmbeddings
from llama_index.utils.oracleai import OracleSummary
from llama_index.vector_stores.oracledb import OraLlamaVS, DistanceStrategy
from llama_index.vector_stores.oracledb import base as orallamavs

if TYPE_CHECKING:
    import oracledb

"""
In this sample example, we will use 'database' provider for both summary and embeddings.
So, we don't need to do the following:
    - set proxy for 3rd party providers
    - create credential for 3rd party providers

If you choose to use 3rd party provider,
please follow the necessary steps for proxy and credential.
"""

# oracle connection
# please update with your username, password, hostname, and service_name
username = "testuser"
password = "testuser"
dsn = "<hostname/service_name>"

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")
    sys.exit(1)


# load onnx model
# please update with your related information
onnx_dir = "DEMO_PY_DIR"
onnx_file = "tinybert.onnx"
model_name = "demo_model"
try:
    OracleEmbeddings.load_onnx_model(conn, onnx_dir, onnx_file, model_name)
    print("ONNX model loaded.")
except Exception as e:
    print("ONNX model loading failed!")
    sys.exit(1)


# params
# please update necessary fields with related information
loader_params = {
    "owner": "testuser",
    "tablename": "demo_tab",
    "colname": "data",
}
summary_params = {
    "provider": "database",
    "glevel": "S",
    "numParagraphs": 1,
    "language": "english",
}
splitter_params = {"normalize": "all"}
embedder_params = {"provider": "database", "model": "demo_model"}

# instantiate loader, summary, splitter, and embedder
loader = OracleReader(conn=conn, params=loader_params)
summary = OracleSummary(conn=conn, params=summary_params)
splitter = OracleTextSplitter(conn=conn, params=splitter_params)
embedder = OracleEmbeddings(conn=conn, params=embedder_params)

# process the documents
loader = OracleReader(conn=conn, params=loader_params)
docs = loader.load()

chunks_with_mdata = []
for id, doc in enumerate(docs, start=1):
    summ = summary.get_summary(doc.text)
    chunks = splitter.split_text(doc.text)
    for ic, chunk in enumerate(chunks, start=1):
        chunk_metadata = doc.metadata.copy()
        chunk_metadata["id"] = (
            chunk_metadata["_oid"] + "$" + str(id) + "$" + str(ic)
        )
        chunk_metadata["document_id"] = str(id)
        chunk_metadata["document_summary"] = str(summ[0])
        textnode = TextNode(
            text=chunk,
            id_=chunk_metadata["id"],
            embedding=embedder._get_text_embedding(chunk),
            metadata=chunk_metadata,
        )
        chunks_with_mdata.append(textnode)

""" verify """
print(f"Number of total chunks with metadata: {len(chunks_with_mdata)}")


# create Oracle AI Vector Store
vectorstore = OraLlamaVS.from_documents(
    client=conn,
    docs=chunks_with_mdata,
    table_name="oravs",
    distance_strategy=DistanceStrategy.DOT_PRODUCT,
)

""" verify """
print(f"Vector Store Table: {vectorstore.table_name}")

# Create Index
orallamavs.create_index(
    conn, vectorstore, params={"idx_name": "hnsw_oravs", "idx_type": "HNSW"}
)

print("Index created.")


# Perform Semantic Search
query = "What is Oracle AI Vector Store?"
filter = {"document_id": ["1"]}

# Similarity search without a filter
print(vectorstore.similarity_search(query, 1))

# Similarity search with a filter
print(vectorstore.similarity_search(query, 1, filter=filter))

# Similarity search with relevance score
print(vectorstore.similarity_search_with_score(query, 1))

# Similarity search with relevance score with filter
print(vectorstore.similarity_search_with_score(query, 1, filter=filter))

# Max marginal relevance search
print(
    vectorstore.max_marginal_relevance_search(
        query, 1, fetch_k=20, lambda_mult=0.5
    )
)

# Max marginal relevance search with filter
print(
    vectorstore.max_marginal_relevance_search(
        query, 1, fetch_k=20, lambda_mult=0.5, filter=filter
    )
)
```
