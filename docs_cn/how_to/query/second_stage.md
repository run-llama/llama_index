二级处理

默认情况下，当在索引或组合图上执行查询时，LlamaIndex执行以下步骤：
1. **检索步骤**：根据查询从索引中检索一组节点。
2. **综合步骤**：在节点集上综合响应。

除了标准检索和综合之外，LlamaIndex还提供了一系列用于高级**二级处理**（即检索和综合之后）的模块。

在检索初始候选节点后，这些模块通过例如过滤、重新排序或增强来进一步提高用于综合的节点的质量和多样性。示例包括关键字过滤器、基于LLM的重新排序和基于时间推理的增强。

我们首先提供高级API接口，然后提供一些示例模块，最后讨论使用情况。

我们也非常欢迎贡献！如果您有兴趣贡献Postprocessor，请查看我们的[贡献指南](https://github.com/jerryjliu/llama_index/blob/main/CONTRIBUTING.md)。

## API接口

基类是`BaseNodePostprocessor`，API接口非常简单：

```python

class BaseNodePostprocessor:
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
```

它接受一个节点对象列表，并输出另一个节点对象列表。

完整的API参考文档可以在[这里](/reference/node_postprocessor.rst)找到。


## 示例用法

Postprocessor可以作为`ResponseSynthesizer`的一部分在`QueryEngine`中使用，也可以单独使用。

#### 索引查询

```python

from llama_index.indices.postprocessor import (
    FixedRecencyPostprocessor,
)
node_postprocessor = FixedRecencyPostprocessor(service_context=service_context)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[node_postprocessor]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband (Julian) for Viaweb?", 
)

```


#### 作为独立模块使用（低级用法）

该模块也可以单独使用，作为更广泛流程的一部分。例如，这里有一个示例，您可以手动处理初始源节点集。

```python
from llama_index.indices.postprocessor import (
    FixedRecencyPostprocessor,
)

# 从向量索引获取初始响应
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="no_text"
)
init_response = query_engine.query(query_str)
resp_nodes = [n.node for n in init_response.source_nodes]

# 使用节点后处理器过滤节点
node_postprocessor = FixedRecencyPostprocessor(service_context=service_context)
new_nodes = node_postprocessor.postprocess_nodes(resp_nodes)

# 使用列表索引合成答案
list_index = GPTListIndex(new_nodes)
query_engine = list_index.as_query_engine(
    node_postprocessors=[node_postprocessor]
)
response = query_engine.query(query_str)**`TimeWeightedPostprocessor`**：使用公式`(1-time_decay) ** hours_passed`为检索到的节点添加时间加权。新鲜度得分将添加到节点已有的任何得分中。