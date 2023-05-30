Graphsignal提供AI代理和LLM驱动的应用程序的可观察性。它帮助开发人员确保AI应用程序按预期运行，用户拥有最佳体验。
Graphsignal自动跟踪和监控LlamaIndex。跟踪和指标提供查询、检索和索引操作的执行详细信息。这些见解包括提示、完成、嵌入统计、检索节点、参数、延迟和异常。
当使用OpenAI API时，Graphsignal还提供其他见解，如每次部署、模型或任何上下文的令牌计数和成本。

安装和设置
添加Graphsignal跟踪器很简单，只需安装和配置：

```sh
pip install graphsignal
```

```python
import graphsignal

# 直接提供API密钥或通过GRAPHSIGNAL_API_KEY环境变量
graphsignal.configure(api_key='my-api-key', deployment='my-llama-index-app-prod')
```

您可以在[这里](https://app.graphsignal.com/)获取API密钥。

有关更多信息，请参阅[快速入门指南](https://graphsignal.com/docs/guides/quick-start/)、[集成指南](https://graphsignal.com/docs/integrations/llama-index/)和[示例应用程序](https://github.com/graphsignal/examples/blob/main/llama-index-app/main.py)。

跟踪其他功能
要额外跟踪任何函数或代码，您可以使用装饰器或上下文管理器：

```python
with graphsignal.start_trace('load-external-data'):
    reader.load_data()
```

有关完整说明，请参阅[Python API参考](https://graphsignal.com/docs/reference/python-api/)。

有用的链接
* [跟踪和监控LlamaIndex应用程序](https://graphsignal.com/blog/tracing-and-monitoring-llama-index-applications/)
* [监控OpenAI API延迟、令牌、速率限制等](https://graphsignal.com/blog/monitor-open-ai-api-latency-tokens-rate-limits-and-more/)
* [OpenAI API成本跟踪：按模型、部署和上下文分析费用](https://graphsignal.com/blog/open-ai-api-cost-tracking-analyzing-expenses-by-model-deployment-and-context/)

Graphsignal提供AI代理和LLM驱动的应用程序的可观察性。它帮助开发人员确保AI应用程序按预期运行，用户拥有最佳体验。Graphsignal自动跟踪和监控LlamaIndex，提供查询、检索和索引操作的执行详细信息，包括提示、完成、嵌入统计、检索节点、参数、延迟和异常。当使用OpenAI API时，Graphsignal还提供每次部署、模型或任何上下文的令牌计数和成本等其他见解。添加Graphsignal跟踪器很简单，只需安装和配置，您可以在[这里](https://app.graphsignal.com/)获取API密钥。要额外跟踪任何函数或代码，您可以使用装饰器或上下文管理器。有关更多信息，请参阅[快速入门指南](https://graphsignal.com/docs/guides/quick-start/)、[集成指南](https://graphsignal.com/docs/integrations/llama-index/)和[示例应用程序](https://github.com/graphsignal/examples/blob/main/llama-index-app/main.py)，以及[Python API参考](https://graphsignal.com/docs/reference/python-api/)。此外，还可以参考[跟踪和监控LlamaIndex应用程序](https://graphsignal.com/blog/tracing-and-monitoring-llama-index-applications/)、[监控OpenAI API延迟、令牌、速率限制等](https://graphsignal.com/blog/monitor-open-ai-api-latency-tokens-rate-limits-and-more/)和[OpenAI API成本跟踪：按模型、部署和上下文分析费用](https://graphsignal.com/blog/open-ai-api-cost-tracking-analyzing-expenses-by-model-deployment-and-context/)等相关文章。