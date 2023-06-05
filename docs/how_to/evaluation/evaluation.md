# 🔬 Evaluation

LlamaIndex offers a few key modules for evaluating the quality of both Document retrieval and response synthesis.
Here are some key questions for each component:
- **Document retrieval**: Are the sources relevant to the query?
- **Response synthesis**: Does the response match the retrieved context? Does it also match the query? 

This guide describes how the evaluation components within LlamaIndex work. Note that our current evaluation modules
do *not* require ground-truth labels. Evaluation can be done with some combination of the query, context, response,
and combine these with LLM calls.

## Evaluation of the Response + Context

Each response from an `query_engine.query` calls returns both the synthesized response as well as source documents.

We can evaluate the response against the retrieved sources - without taking into account the query!

This allows you to measure hallucination - if the response does not match the retrieved sources, this means that the model may be "hallucinating" an answer
since it is not rooting the answer in the context provided to it in the prompt.

There are two sub-modes of evaluation here. We can either get a binary response "YES"/"NO" on whether response matches *any* source context,
and also get a response list across sources to see which sources match.

### Binary Evaluation

This mode of evaluation will return "YES"/"NO" if the synthesized response matches any source context.

```python
from llama_index import VectorStoreIndex
from llama_index.evaluation import ResponseEvaluator

# build service context
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

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

#### Diagram

![](/_static/evaluation/eval_response_context.png)




### Sources Evaluation

This mode of evaluation will return "YES"/"NO" for every source node.

```python
from llama_index import VectorStoreIndex
from llama_index.evaluation import ResponseEvaluator

# build service context
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

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

### Notebook

Take a look at this [notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/evaluation/TestNYC-Evaluation.ipynb).


## Evaluation of the Query + Response + Source Context

This is similar to the above section, except now we also take into account the query. The goal is to determine if
the response + source context answers the query.

As with the above, there are two submodes of evaluation. 
- We can either get a binary response "YES"/"NO" on whether
the response matches the query, and whether any source node also matches the query.
- We can also ignore the synthesized response, and check every source node to see
if it matches the query.

### Binary Evaluation

This mode of evaluation will return "YES"/"NO" if the synthesized response matches the query + any source context.

```python
from llama_index import VectorStoreIndex
from llama_index.evaluation import QueryResponseEvaluator

# build service context
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# build index
...

# define evaluator
evaluator = QueryResponseEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query("What battles took place in New York City in the American Revolution?")
eval_result = evaluator.evaluate(response)
print(str(eval_result))

```

#### Diagram

![](/_static/evaluation/eval_query_response_context.png)


### Sources Evaluation

This mode of evaluation will look at each source node, and see if each source node contains an answer to the query.

```python
from llama_index import VectorStoreIndex
from llama_index.evaluation import QueryResponseEvaluator

# build service context
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# build index
...

# define evaluator
evaluator = QueryResponseEvaluator(service_context=service_context)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query("What battles took place in New York City in the American Revolution?")
eval_result = evaluator.evaluate_source_nodes(response)
print(str(eval_result))

```

#### Diagram

![](/_static/evaluation/eval_query_sources.png)

### Notebook

Take a look at this [notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/evaluation/TestNYC-Evaluation-Query.ipynb).


```{toctree}
---
caption: Examples
maxdepth: 1
---

../../examples/evaluation/TestNYC-Evaluation.ipynb
../../examples/evaluation/TestNYC-Evaluation-Query.ipynb
../../examples/evaluation/QuestionGeneration.ipynb
```