# Usage Pattern

## Evaluating Response for Hallucination

### Binary Evaluation

This mode of evaluation will return "YES"/"NO" if the synthesized response matches any source context.

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import ResponseEvaluator

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

# build index
...

# define evaluator
evaluator = ResponseEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query("What battles took place in New York City in the American Revolution?")
eval_result = evaluator.evaluate(response)
print(str(eval_result))

```

You'll get back either a `YES` or `NO` response.

![](/_static/evaluation/eval_response_context.png)

### Sources Evaluation

This mode of evaluation will return "YES"/"NO" for every source node.

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import ResponseEvaluator

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

# build index
...

# define evaluator
evaluator = ResponseEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query("What battles took place in New York City in the American Revolution?")
eval_result = evaluator.evaluate_source_nodes(response)
print(str(eval_result))

```

You'll get back a list of "YES"/"NO", corresponding to each source node in `response.source_nodes`.

## Evaluting Query + Response for Answer Quality

### Binary Evaluation

This mode of evaluation will return "YES"/"NO" if the synthesized response matches the query + any source context.

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import QueryResponseEvaluator

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

# build index
...

# define evaluator
evaluator = QueryResponseEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
query = "What battles took place in New York City in the American Revolution?"
response = query_engine.query(query)
eval_result = evaluator.evaluate(query, response)
print(str(eval_result))

```

![](/_static/evaluation/eval_query_response_context.png)

### Sources Evaluation

This mode of evaluation will look at each source node, and see if each source node contains an answer to the query.

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import QueryResponseEvaluator

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

# build index
...

# define evaluator
evaluator = QueryResponseEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
query = "What battles took place in New York City in the American Revolution?"
response = query_engine.query(query)
eval_result = evaluator.evaluate_source_nodes(query, response)
print(str(eval_result))
```

![](/_static/evaluation/eval_query_sources.png)

## Question Generation

LlamaIndex can also generate questions to answer using your data. Using in combination with the above evaluators, you can create a fully automated evaluation pipeline over your data.

```python
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import ResponseEvaluator

# build service context
llm = OpenAI(model="gpt-4", temperature=0.0)
service_context = ServiceContext.from_defaults(llm=llm)

# build documents
documents = SimpleDirectoryReader("./data").load_data()

# define genertor, generate questions
data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()
```

## Integrations

We also integrate with community evaluation tools.

- [DeepEval](../../../community/integrations/deepeval.md)
- [Ragas](https://github.com/explodinggradients/ragas/blob/main/docs/integrations/llamaindex.ipynb)
