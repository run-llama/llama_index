成本分析
每次调用LLM都会花费一定的金额 - 例如，OpenAI的Davinci的费用为每1k令牌0.02美元。构建索引和查询的成本取决于
- 所使用的LLM类型
- 所使用的数据结构类型
- 构建期间使用的参数
- 查询期间使用的参数

构建和查询每个索引的成本是参考文档中的TODO。与此同时，我们提供以下信息：

1. 索引成本结构的高级概述。
2. 您可以在LlamaIndex中直接使用的令牌预测器！

### 成本结构概述

#### 不需要LLM调用的索引
以下索引在构建期间根本不需要LLM调用（0成本）：
- `GPTListIndex`
- `GPTSimpleKeywordTableIndex` - 使用正则表达式关键字提取器从每个文档中提取关键字
- `GPTRAKEKeywordTableIndex` - 使用RAKE关键字提取器从每个文档中提取关键字

#### 需要LLM调用的索引
以下索引确实需要在构建期间调用LLM：
- `GPTTreeIndex` - 使用LLM分层汇总文本以构建树
- `GPTKeywordTableIndex` - 使用LLM从每个文档中提取关键字

### 查询时间

查询时总是会有> = 1个LLM调用，以合成最终答案。
某些索引在索引构建和查询之间存在成本权衡。例如，`GPTListIndex`
免费构建，但在列表索引上运行查询（无过滤或嵌入查找）将
调用LLM {math}`N`次。

以下是关于每个索引的一些注释：
- `GPTListIndex`：默认需要{math}`N`次LLM调用，其中N是节点数。
- `GPTTreeIndex`：默认需要{math}`\log（N）`次LLM调用，其中N是叶节点数。
    - 设置`child_branch_factor = 2`将比默认的`child_branch_factor = 1`更昂贵（多项式vs对数），因为我们为每个父节点遍历2个子节点而不是1个。
- `GPTKeywordTableIndex`：默认需要LLM调用以提取查询关键字。
    - 可以执行`index.as_retriever（retriever_mode =“simple”）`或`index.as_retriever（retriever_mode =“rake”）`以在查询文本上也使用正则表达式/ RAKE关键字提取器。

### 令牌预测器使用

LlamaIndex提供令牌**预测器**来预测LLM和嵌入调用的令牌使用情况。
这样，您可以在1）索引构建和2）索引查询之前，估计相应的LLM调用的成本。

#### 使用MockLLMPredictor

要预测LLM调用的令牌使用情况，请导入并实例化MockLL您可以在索引构建和查询期间使用以下MockLLMPredictor：
```python
from llama_index import MockLLMPredictor, ServiceContext

llm_predictor = MockLLMPredictor(max_tokens=256)
```
然后，您可以在以下示例中使用此预测器。

**索引构建**
```python
from llama_index import GPTTreeIndex, MockLLMPredictor, SimpleDirectoryReader

documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
# “模拟” llm 预测器是我们的令牌计数器
llm_predictor = MockLLMPredictor(max_tokens=256)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
# 在索引构建期间将“模拟” llm_predictor传递到GPTTreeIndex
index = GPTTreeIndex.from_documents(documents, service_context=service_context)

# 获取使用的令牌数
print(llm_predictor.last_token_usage)
```

**索引查询**

```python
query_engine = index.as_query_engine(
    service_context=service_context
)
response = query_engine.query("What did the author do growing up?")

# 获取使用的令牌数
print(llm_predictor.last_token_usage)
```

#### 使用MockEmbedding

您还可以使用`MockEmbedding`预测嵌入调用的令牌使用量。您可以将其与`MockLLMPredictor`一起使用。

```python
from llama_index import (
    GPTVectorStoreIndex,
    MockLLMPredictor,
    MockEmbedding,
    SimpleDirectoryReader,
    ServiceContext
)

documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# 指定MockLLMPredictor以及MockEmbedding
llm_predictor = MockLLMPredictor(max_tokens=256)
embed_model = MockEmbedding(embed_dim=1536)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

query_engine = index.as_query_engine(
    service_context=service_context
)
response = query_engine.query(
    "What did the author do after his time at Y Combinator?",
)
```

[这里有一个示例笔记本](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/analysis/TokenPredictor.ipynb)。

```{toctree}
---
caption: 示例
maxdepth: 1
---
../../examples/analysis/TokenPredictor.ipynb
```
