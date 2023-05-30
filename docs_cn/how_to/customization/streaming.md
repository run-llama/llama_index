LlamaIndex支持在生成响应的同时流式传输响应。这使您可以在完整响应完成之前开始打印或处理响应的开头。这可以大大降低查询的感知延迟。

### 设置
要启用流式传输，您需要配置两件事：
1.使用支持流式传输的LLM，并设置“streaming = True”。
```python
llm_predictor = LLMPredictor(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor
)
```
目前，OpenAI和HuggingFace LLMs支持流式传输。

2.配置查询引擎以使用流式传输

如果您使用高级API，在构建查询引擎时设置“streaming = True”。
```python
query_engine = index.as_query_engine(
    streaming=True,
    similarity_top_k=1
)
```

如果您使用低级API组合查询引擎，请在构造“ResponseSynthesizer”时传递“streaming = True”：
```python
synth = ResponseSynthesizer.from_args(streaming=True, ...)
query_engine = RetrieverQueryEngine(response_synthesizer=synth, ...)
```

### 流式响应
在正确配置LLM和查询引擎后，调用“query”现在会返回一个“StreamingResponse”对象。

```python
streaming_response = query_engine.query(
    "What did the author do growing up?", 
)
```

LLM调用*开始*时，响应立即返回，无需等待完整完成。

>注意：在查询引擎进行多次LLM调用的情况下，只有最后一次LLM调用才会被流式传输，并且响应在最后一次LLM调用开始时返回。

您可以从流式响应中获取一个“Generator”，并在他们到达时迭代令牌：
```python
for text in streaming_response.response_gen:
    # do something with text as they arrive.
```

或者，如果您只想在他们到达时打印文本：
```
streaming_response.print_response_stream() 
```