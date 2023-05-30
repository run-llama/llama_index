定义提示
提示是LLMs的基本输入，提供其表达能力。LlamaIndex使用提示来构建索引，插入，在查询期间执行遍历以及合成最终答案。
LlamaIndex使用一组[默认提示模板](https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/default_prompts.py)，可以直接使用。
用户也可以提供自己的提示模板，以进一步定制框架的行为。

## 定义自定义提示
定义自定义提示只需要创建一个格式字符串即可：

```python
from llama_index import Prompt

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)
qa_template = Prompt(template)
```

> 注意：您可能会看到对遗留提示子类（如`QuestionAnswerPrompt`，`RefinePrompt`）的引用。这些已经被弃用（现在是`Prompt`的类型别名）。现在，您可以直接指定`Prompt（template）`来构造自定义提示。但是，您仍然必须确保模板字符串包含预期的参数（例如`{context_str}`和`{query_str}`），以替换默认的问答提示。

## 将自定义提示传递到管道中
由于LlamaIndex是一个多步管道，因此确定要修改的操作并在正确的位置传入自定义提示是很重要的。
从高层次上讲，提示用于1）索引构建和2）查询引擎执行。

### 修改索引构建中使用的提示
不同的索引在构建过程中使用不同类型的提示（有些根本不使用提示）。
例如，`GPTTreeIndex`使用`SummaryPrompt`来分层汇总节点，`GPTKeywordTableIndex`使用`KeywordExtractPrompt`来提取关键字。

有两种等效的方法可以覆盖提示：
1. 通过默认节点构造器
```python
index = GPTTreeIndex(nodes, summary_template=<custom_prompt>)
```
2. 通过文档构造器。
```python
index = GPTTreeIndex.from_documents(docs, summary_template=<custom_prompt>)
```

有关哪个索引使用哪些提示的更多详细信息，请访问[索引类引用](/reference/indices.rst)。

### 修改查询引擎中使用的提示
更常见的是，提示在查询时使用（即对索引执行查询并合成最终响应）。也有两种等效的方法可以覆盖提示：
1. 通过高上面的两种方法是等价的，其中1实际上是2的语法糖，隐藏了底层的复杂性。您可能希望使用1来快速修改一些常见参数，并使用2来获得更细粒度的控制。

有关哪些类使用哪些提示，请访问[查询类引用](/reference/query.rst)。

## 完整示例

可以在[此笔记本](https://github.com/jerryjliu/llama_index/blob/main/examples/paul_graham_essay/TestEssay.ipynb)中找到示例。

下面是相应的片段。我们展示如何为问答定义自定义提示，该提示需要`context_str`和`query_str`字段。提示在查询时间传入。

请参阅[参考文档](/reference/prompts.rst)以获取所有提示的完整集合。