LlamaIndex允许您对索引结构执行*查询转换*。查询转换是将查询转换为另一个查询的模块。它们可以是**单步**的，即在执行查询之前只运行一次转换。
它们也可以是**多步**的，即：
1. 将查询转换，对索引执行
2. 检索响应
3. 以顺序方式转换/执行后续查询。

我们在下面详细列出了一些查询转换。

#### 用例
查询转换有多种用例：
- 将初始查询转换为更容易嵌入的形式（例如HyDE）
- 将初始查询转换为可以更容易从数据中获得答案的子问题（单步查询分解）
- 将初始查询分解为多个子问题，这些子问题可以更容易地单独解决。 （多步查询分解）

### HyDE（假设文档嵌入）

[HyDE](http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf)是一种技术，给定自然语言查询，首先生成假设文档/答案。然后使用这个假设文档进行嵌入查找，而不是原始查询。

要使用HyDE，下面显示了一个示例代码片段。

```python
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.indices.query import TransformQueryEngine

# 加载文档，构建索引
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex(documents)

# 使用HyDE查询转换运行查询
query_str = "what did paul graham do after going to RISD"
hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
response = query_engine.query(query_str)
print(response)

```

请查看我们的[示例笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/query_transformations/HyDEQueryTransformDemo.ipynb)以获得完整的指导。

### 单步查询分解

最近的一些方法（例如[self-ask](https://ofir.io/self-ask.pdf)，[ReAct](https://arxiv.org/abs/2210.03629)）建议，当LLM将问题分解为较小的步骤时，它们在回答复杂问题时表现更好。我们发现，对于需要知识增强的查询也是如此。

如果您的查询很复杂，不同的部分可以被分解为更容易回答的子问题，并且可以使用查询转换来实现。您的知识库可以回答围绕整个查询的不同“子查询”。

我们的单步查询分解功能将复杂的问题转换为数据集上的简单问题，以帮助提供对原始问题的子答案。

这在[组合图](/how_to/index_structs/composability.md)上尤其有用。在组合图中，查询可以路由到多个子索引，每个子索引代表整个知识语料库的一个子集。查询分解允许我们将查询转换为任何给定索引上的更合适的问题。

下面是一个示例图片。

![](/_static/query_transformations/single_step_diagram.png)

以下是在组合图上的相应示例代码片段。

```python

# 设置：由多个向量索引组成的列表索引
# llm_predictor_chatgpt对应于ChatGPT LLM接口
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)

# 初始化索引和图
...


# 配置检索器
vector_query_engine = vector_index.as_query_engine()
vector_query_engine = TransformQueryEngine(
    vector_query_engine, 
    query_transform=decompose_transform
    transform_extra_info={'index_summary': vector_index.index_struct.summary}
)
custom_query_engines = {
    vector_index.index_id: vector_query_engine
} 

# 查询
query_str = (
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
response = query_engine.query(query_str)
```

请查看我们的[示例笔记本](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/composable_indices/city_analysis/City_Analysis-Decompose.ipynb)以获取完整的指南。

### 多步查询转换

多步查询转换是基于现有的单步查询转换方法的概括。

给定初始的复杂查询，将对索引进行查询转换和执行。从查询中检索响应。
给定响应（以及先前的响应）和查询，也可以对索引提出后续问题。这种技术允许将查询运行到单个知识源，直到查询满足所有问题为止。

下面是一个示例图片。

![](/_static/query_transformations/multi_step_diagram.png)


以下是相应的示例代码片段。

```python
from llama_index.indices.query.query_transform.bas
```我们提供了一个[示例笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/vector_indices/SimpleIndexDemo-multistep.ipynb)，您可以查看完整的步骤。使用StepDecomposeQueryTransform，我们可以创建一个MultiStepQueryEngine，它可以接受一个查询，例如“作者开始的加速器计划的第一批是谁？”，并返回一个响应。