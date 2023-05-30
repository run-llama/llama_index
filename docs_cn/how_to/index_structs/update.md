更新索引
每个LlamaIndex数据结构都允许**插入**、**删除**和**更新**。

### 插入

您可以在最初构建索引后，将新文档“插入”到任何索引数据结构中。插入背后的机制取决于索引结构。
例如，对于列表索引，新文档将作为列表中的附加节点插入。
对于向量存储索引，新文档（和嵌入）将插入到底层文档/嵌入存储中。

[这里](https://github.com/jerryjliu/llama_index/blob/main/examples/paul_graham_essay/InsertDemo.ipynb)给出了一个展示我们插入功能的示例笔记本。
在这个笔记本中，我们展示了如何构建一个空索引，手动创建文档对象，并将这些对象添加到我们的索引数据结构中。

下面给出了一个示例代码片段：

```python
index = GPTListIndex([])

embed_model = OpenAIEmbedding()
doc_chunks = []
for i, text in enumerate(text_chunks):
    doc = Document(text, doc_id=f"doc_id_{i}")
    doc_chunks.append(doc)

# insert
for doc_chunk in doc_chunks:
    index.insert(doc_chunk)

```

### 删除

您可以通过指定document_id“删除”大多数索引数据结构中的文档。（**注意**：树索引目前不支持删除）。所有与
文档相关的节点都将被删除。

**注意**：要删除文档，必须在首次加载到索引中时指定文档的doc_id。

```python
index.delete("doc_id_0")
```


### 更新

如果文档已经存在于索引中，则可以使用相同的`doc_id`“更新”文档（例如，如果文档中的信息已更改）。

```python
# NOTE: the document has a `doc_id` specified
index.update(doc_chunks[0])
```