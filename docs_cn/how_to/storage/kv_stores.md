键值存储是支持我们的[文档存储](/how_to/storage/docstores.md)和[索引存储](/how_to/storage/index_stores.md)的底层存储抽象。

我们提供以下键值存储：
- **简单键值存储**：一个内存中的KV存储。用户可以选择在此KV存储上调用`persist`以将数据持久化到磁盘。
- **MongoDB键值存储**：一个MongoDB KV存储。

有关更多详细信息，请参阅[API参考](/reference/storage/kv_store.rst)。

注意：目前，这些存储抽象不是外部面向的。