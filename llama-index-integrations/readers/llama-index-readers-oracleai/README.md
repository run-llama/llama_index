# LlamaIndex Readers Integration: Oracleai

There are two classes here:

- OracleReader: This API is to load document(s) from a file or a directory or a Oracle Database table.
- OracleTextSplitter: This API is to split a document into chunks with a lots of customizations.

`pip install llama-index-readers-oracleai`

# A sample example

```python
# get the Oracle connection
conn = oracledb.connect(
    user="",
    password="",
    dsn="",
)
print("Oracle connection is established...")

# params
loader_params = {"owner": "ut", "tablename": "demo_tab", "colname": "data"}
splitter_params = {"by": "words", "max": "100"}

# instances
loader = OracleReader(conn=conn, params=loader_params)
splitter = OracleTextSplitter(conn=conn, params=splitter_params)

print("Processing the documents...")
docs = loader.load()
for id, doc in enumerate(docs, start=1):
    print(f"Document#{id}, Metadata: {doc.metadata}")
    chunks = splitter.split_text(doc.text)
    print(f"Document#{id}, Num of Chunk: {len(chunks)}\n")

conn.close()
print("Connection is closed.")
```
