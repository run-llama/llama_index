文档存储
文档存储包含已摄入的文档块，我们称之为“节点”对象。
有关更多详细信息，请参阅[API参考](/reference/storage/docstore.rst)。

### 简单文档存储
默认情况下，`SimpleDocumentStore`将`Node`对象存储在内存中。
可以通过调用`docstore.persist()`（和`SimpleDocumentStore.from_persist_path(...)`）将它们持久化到（和从）磁盘。

### MongoDB文档存储
我们支持MongoDB作为替代文档存储后端，在摄入`Node`对象时将数据持久化。
```python
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.node_parser import SimpleNodeParser

# 创建解析器并将文档解析为节点
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# 创建（或加载）docstore并添加节点
docstore = MongoDocumentStore.from_uri(uri="<mongodb+srv://...>")
docstore.add_documents(nodes)

# 创建存储上下文
storage_context = StorageContext.from_defaults(docstore=docstore)

# 构建索引
index = GPTVectorStoreIndex(nodes, storage_context=storage_context)
```

在底层，`MongoDocumentStore`连接到固定的MongoDB数据库，并为您的节点初始化新的集合（或加载现有集合）。
> 注意：在实例化`MongoDocumentStore`时，您可以配置`db_name`和`namespace`，否则它们默认为`db_name="db_docstore"`和`namespace="docstore"`。

请注意，使用`MongoDocumentStore`时不需要调用`storage_context.persist()`（或`docstore.persist()`），因为数据默认情况下会被持久化。

您可以通过重新使用现有的`db_name`和`collection_name`来重新连接到MongoDB集合并重新加载索引。