LlamaIndex提供了您的索引的可组合性，这意味着您可以在其他索引之上构建索引。这样可以更有效地索引整个文档树，以便为GPT提供自定义知识。

可组合性允许您为每个文档定义低级索引，以及在文档集合上定义高阶索引。要了解这是如何工作的，请想象为文本定义树索引，以及在您的集合中的每个树索引（一个文档）上定义列表索引。

###定义子索引
要了解这是如何工作的，请想象您有3个文档：`doc1`，`doc2`和`doc3`。

现在让我们为每个文档定义一个树索引。为了稍后持久化图形，每个索引应共享相同的存储上下文。

在Python中，我们有：

###定义摘要文本

然后，您需要为每个子索引显式定义*摘要文本*。这样，子索引就可以用作高级索引的文档。

您可以选择手动指定摘要文本，或者使用LlamaIndex本身生成摘要，例如：

###创建具有顶级索引的图

然后，我们可以在这3个树索引之上创建一个具有列表索引的图：我们可以像其他索引一样查询，保存和加载图形。查询图
在查询期间，我们将从顶级列表索引开始。列表中的每个节点对应于底层树索引。查询将从根索引开始递归执行，然后是子索引。每个索引的默认查询引擎称为背后（即`index.as_query_engine（）`），除非通过将`custom_query_engines`传递给`ComposableGraphQueryEngine`进行其他配置。下面我们展示一个将树索引检索器配置为使用`child_branch_factor=2`（而不是默认的`child_branch_factor=1`）的示例。
更多有关如何配置`ComposableGraphQueryEngine`的详细信息，请参阅[此处](/reference/query/query_engines/graph_query_engine.rst)。
在节点内，我们将递归查询存储的树索引以检索答案，而不是获取文本。
注意：按索引ID为索引指定自定义检索器可能需要您检查例如`index1.index_id`。或者，您可以显式地将其设置如下：
```python
index1.set_index_id（“<index_id_1>”）
index2.set_index_id（“<index_id_2>”）
index3.set_index_id（“<index_id_3>”）
```
您可以根据您的知识库的层次结构堆叠指数，尽可能多地堆叠指数！
[可选]持久化图
图也可以持久化到存储，然后在需要时再次加载。请注意，您需要设置根索引的ID，或者跟踪默认值。
我们可以看一下代码示例。我们首先构建两个树索引，一个是维基百科的纽约市页面，另一个是保罗·格雷厄姆的文章。然后，我们定义一个关键字提取索引，覆盖这两个树索引。
[这里有一个示例笔记本](https://github.com/jerryjliu/llama_index)---
标题：示例
最大深度：1
---
../../examples/composable_indices/ComposableIndices-Prior.ipynb
../../examples/composable_indices/ComposableIndices-Weaviate.ipynb
../../examples/composable_indices/ComposableIndices.ipynb