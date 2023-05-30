索引存储包含轻量级索引元数据（即在构建索引时创建的附加状态信息）。有关详细信息，请参阅[API参考](/reference/storage/index_store.rst)。

### 简单索引存储
默认情况下，LlamaIndex使用基于内存键值存储的简单索引存储。它们可以通过调用`index_store.persist()`（和`SimpleIndexStore.from_persist_path(...)`）持久化到（和从）磁盘。

### MongoDB索引存储
与文档存储类似，我们还可以使用`MongoDB`作为索引存储的存储后端。

```python
from llama_index.storage.index_store import MongoIndexStore


# create (or load) index store
index_store = MongoIndexStore.from_uri(uri="<mongodb+srv://...>")

# create storage context
storage_context = StorageContext.from_defaults(index_store=index_store)

# build index
index = GPTVectorStoreIndex(nodes, storage_context=storage_context)

# or alternatively, load index
index = load_index_from_storage(storage_context)
```

在底层，`MongoIndexStore`连接到固定的MongoDB数据库，并为您的索引元数据初始化新的集合（或加载现有集合）。
> 注意：您可以在实例化`MongoIndexStore`时配置`db_name`和`namespace`，否则它们默认为`db_name="db_docstore"`和`namespace="docstore"`。

请注意，使用`MongoIndexStore`时不需要调用`storage_context.persist()`（或`index_store.persist()`），因为数据默认情况下会被持久化。

您可以通过使用现有的`db_name`和`collection_name`重新初始化`MongoIndexStore`来轻松重新连接到MongoDB集合并重新加载索引。