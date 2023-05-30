持久化和加载数据

## 持久化数据
默认情况下，LlamaIndex将数据存储在内存中，如果需要，可以显式持久化数据：
```python
storage_context.persist(persist_dir="<persist_dir>")
```
这将把数据持久化到磁盘上，在指定的`persist_dir`（或默认的`./storage`）下。

可以从同一个目录持久化和加载多个索引，只要你跟踪索引ID以便加载。

用户还可以配置替代存储后端（例如`MongoDB`），它们默认情况下会持久化数据。在这种情况下，调用`storage_context.persist()`将不起作用。

## 加载数据
要加载数据，用户只需使用相同的配置重新创建存储上下文（例如传入相同的`persist_dir`或向量存储客户端）即可。

```python
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="<persist_dir>"),
    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="<persist_dir>"),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="<persist_dir>"),
)
```

然后，我们可以通过以下一些便利函数从`StorageContext`中加载特定的索引。

```python
from llama_index import load_index_from_storage, load_indices_from_storage, load_graph_from_storage

# 加载单个索引
# 如果多个索引被持久化到同一个目录，则需要指定索引ID
index = load_index_from_storage(storage_context, index_id="<index_id>") 

# 如果存储上下文中只有一个索引，则不需要指定索引ID
index = load_index_from_storage(storage_context) 

# 加载多个索引
indices = load_indices_from_storage(storage_context) # 加载所有索引
indices = load_indices_from_storage(storage_context, index_ids=[index_id1, ...]) # 加载特定索引

# 加载可组合图
graph = load_graph_from_storage(storage_context, root_id="<root_id>") # 加载指定root_id的图
```

以下是关于保存和加载的[完整API参考](/reference/storage/indices_save_load.rst)。

## 使用远程后端

默认情况下，LlamaIndex使用本地文件系统加载和保存文件。但是，您可以通过传递`fsspec.AbstractFileSystem`对象来覆盖此操作。

这里有一个简单的示例，实例化一个向量存储：
```python
import dotenv
import s3fs
import os
dotenv.load_dotenv("../../../.env")

# 加载文档
documents = SimpleDirectoryReader('../../../examples/paul_graham_essay/data/').load_data()
print(len(documents))
index = GPTVectorStoreIndex.from_documents(documents)
```

到这一步，一切都是一样的。现在 - 让我们实例化一个S3文件系统并从那里保存/加载。

```python
# 设置s3fs
AWS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET = os.environ['AWS_SECRET_ACCESS_KEY']
R2_ACCOUNT_ID = os.environ['R2_ACCOUNT_ID']

assert AWS_KEY is not None and AWS_KEY != ""

s3 = s3fs.S3FileSystem(
   key=AWS_KEY,
   secret=AWS_SECRET,
   endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
   s3_additional_kwargs={'ACL': 'public-read'}
)

# 将索引保存到远程blob存储
index.set_index_id("vector_index")
# 这是{bucket_name}/{index_name}
index.storage_context.persist('llama-index/storage_demo', fs=s3)

# 从s3加载索引
sc = StorageContext.from_defaults(persist_dir='llama-index/storage_demo', fs=s3)
index2 = load_index_from_storage(sc, 'vector_index')
```

默认情况下，如果您不传递文件系统，我们将假定为本地文件系统。