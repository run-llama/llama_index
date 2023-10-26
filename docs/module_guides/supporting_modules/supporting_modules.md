# Supporting Modules

We have two configuration modules that can be configured separately and passed to individual indexes, or set globally.

- The [ServiceContext](service_context.md) includes the LLM you're using, the embedding model, your node parser, your callback manager and more.
- The `StorageContext` lets you specify where and how to store your documents, your vector embeddings, and your indexes. To learn more, read about [customizing storage](/module_guides/storing/customization.md)

```{toctree}
---
maxdepth: 1
hidden: true
---
service_context.md
```
