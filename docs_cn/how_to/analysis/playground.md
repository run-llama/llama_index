Playground模块是LlamaIndex中的一种自动测试您的数据（即文档）的方法，可以在多种索引，模型，嵌入，模式等组合中进行测试，以确定哪些最适合您的目的。将继续添加更多选项。

对于每种组合，您都可以比较任何查询的结果，并比较答案，延迟，使用的令牌等。

您可以使用预设索引从文档列表中初始化一个Playground，或者使用预先构建的索引列表初始化一个Playground。

### 示例代码

以下给出了一个示例用法。

```python
from llama_index import download_loader
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.indices.tree.base import GPTTreeIndex
from llama_index.playground import Playground

# 加载数据
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Berlin'])

# 定义多个索引数据结构（向量索引，列表索引）
indices = [GPTVectorStoreIndex(documents), GPTTreeIndex(documents)]

# 初始化操场
playground = Playground(indices=indices)

# 操场比较
playground.compare("What is the population of Berlin?")

```

### API参考

[API参考文档](/reference/playground.rst)

### 示例笔记本

[示例笔记本链接](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/analysis/PlaygroundDemo.ipynb)。

```{toctree}
---
caption: 示例
maxdepth: 1
---
../../examples/analysis/PlaygroundDemo.ipynb
```