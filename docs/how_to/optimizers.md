# Optimizers

**NOTE**: We'll be adding more to this section soon!

Our optimizers module consists of ways for users to optimize for token usage (we are currently
exploring ways to expand optimization capabilities to other areas, such as performance!)

Here is a sample code snippet on comparing the outputs without optimization and with.

```python
from gpt_index import GPTSimpleVectorIndex
from gpt_index.optimization.optimizer import SentenceEmbeddingOptimizer
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('simple_vector_index.json')

print("Without optimization")
start_time = time.time()
res = index.query("What is the population of Berlin?")
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

print("With optimization")
start_time = time.time()
res = index.query("What is the population of Berlin?", optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5))
end_time = time.time()
print("Total time elapsed: {}".format(end_time - start_time))
print("Answer: {}".format(res))

```

Output:
```text
Without optimization
INFO:root:> [query] Total LLM token usage: 3545 tokens
INFO:root:> [query] Total embedding token usage: 7 tokens
Total time elapsed: 2.8928110599517822
Answer: 
The population of Berlin in 1949 was approximately 2.2 million inhabitants. After the fall of the Berlin Wall in 1989, the population of Berlin increased to approximately 3.7 million inhabitants.

With optimization
INFO:root:> [optimize] Total embedding token usage: 7 tokens
INFO:root:> [query] Total LLM token usage: 1779 tokens
INFO:root:> [query] Total embedding token usage: 7 tokens
Total time elapsed: 2.346346139907837
Answer: 
The population of Berlin is around 4.5 million.
```

Full [example notebook here](https://github.com/jerryjliu/llama_index/blob/main/examples/optimizer/OptimizerDemo.ipynb).

#### API Reference

An API reference can be found [here](/reference/optimizers.rst).
