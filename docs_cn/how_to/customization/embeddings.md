LlamaIndex提供以下格式的嵌入支持：
- 将嵌入添加到Document对象
- 使用向量存储作为底层索引（例如`GPTVectorStoreIndex`）
- 使用嵌入查询我们的列表和树索引。

## 将嵌入添加到Document对象

您可以在构建索引时传入用户指定的嵌入。这样可以让您控制每个Document的嵌入，而不是让我们为您的文本确定嵌入（请参见下文）。

创建Document时只需指定`embedding`字段：

![](/_static/embeddings/doc_example.jpeg)

## 使用向量存储作为底层索引

请参阅我们的[向量存储](/how_to/integrations/vector_stores.md)指南中的相应部分，了解更多详细信息。

## 在列表/树索引中使用嵌入查询模式

LlamaIndex为我们的树和列表索引提供嵌入支持。除了每个节点存储文本外，每个节点还可以选择性地存储嵌入。
在查询时，我们可以使用嵌入来检索节点，然后调用LLM来综合答案。由于使用嵌入（例如使用余弦相似度）进行相似性查找不需要LLM调用，因此嵌入可以作为更便宜的查找机制，而不是使用LLM来遍历节点。

#### 如何生成嵌入？

由于我们在列表和树索引的*查询时*提供嵌入支持，因此嵌入是懒惰生成的，然后缓存（如果在`query(...)`期间指定了`retriever_mode="embedding"`），而不是在索引构建期间生成。
这种设计选择可以避免在索引构建期间为所有文本块生成嵌入。

注意：我们的[基于向量存储的索引](/how_to/integrations/vector_stores.md)在索引构建期间生成嵌入。

#### 嵌入查找
对于列表索引（`GPTListIndex`）：
- 我们遍历列表中的每个节点，并通过嵌入相似性识别出前k个节点。我们使用这些节点来综合答案。
- 有关更多详细信息，请参阅[列表检索器API](/reference/query/retrievers/list.rst)。
- 注意：列表索引的嵌入模式使用与使用我们的`GPTVectorStoreIndex`的使用大致相同；主要
    区别在于何时生成嵌入（列表索引的查询时间vs.简单向量索引的索引构建）。

对于树索引（`GPTTreeIndex`）：
- 我们从根节点开始，并使用嵌入模式查询树索引。通过选择嵌入式相似性的子节点来遍历树。有关更多详细信息，请参阅[树查询API](/reference/query/retrievers/tree.rst)。示例笔记本可在[此处](https://github.com/jerryjliu/llama_index/blob/main/examples/test_wiki/TestNYC_Embeddings.ipynb)找到。LlamaIndex允许您定义自定义嵌入模块。默认情况下，我们使用OpenAI的`text-embedding-ada-002`。您还可以选择从Langchain的[嵌入](https://langchain.readthedocs.io/en/latest/reference/modules/embeddings.html)模块中插入嵌入。我们引入了一个包装类[`LangchainEmbedding`](/reference/service_context/embeddings.rst)，用于集成到LlamaIndex中。