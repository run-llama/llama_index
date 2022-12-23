# Defining LLMs

The goal of GPT Index is to provide a toolkit of data structures that can organize external information in a manner that 
is easily compatible with the prompt limitations of an LLM. Therefore LLMs are always used to construct the final
answer.
Depending on the [type of index](/reference/indices.rst) being used,
LLMs may also be used during index construction, insertion, and query traversal.

GPT Index uses Langchain's [LLM](https://langchain.readthedocs.io/en/latest/modules/llms.html) 
and [LLMChain](https://langchain.readthedocs.io/en/latest/modules/chains.html) module to define
the underlying abstraction. We introduce a wrapper class, 
[`LLMPredictor`](/reference/llm_predictor.rst), for integration into GPT Index.

By default, we use OpenAI's `text-davinci-003` model. But you may choose to customize
the underlying LLM being used.


## Example

An example snippet of customizing the LLM being used is shown below. 
In this example, we use `text-davinci-002` instead of `text-davinci-003`. Note that 
you may plug in any LLM shown on Langchain's 
[LLM](https://langchain.readthedocs.io/en/latest/modules/llms.html) page.


```python

from gpt_index import GPTKeywordTableIndex, SimpleDirectoryReader, LLMPredictor
from langchain import OpenAI

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))

# load index from disk
index = GPTKeywordTableIndex.load_from_disk('index_table.json', llm_predictor=llm_predictor)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```

In this snipet, the index has already been created and saved to disk. We load
the existing index, and swap in a new `LLMPredictor` that is used during query time.