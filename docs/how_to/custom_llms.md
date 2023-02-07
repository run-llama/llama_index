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

We also introduce a [`PromptHelper` class](/reference/prompt_helper.rst), to
allow the user to explicitly set certain constraint parameters, such as 
maximum input size (default is 4096 for davinci models), number of generated output
tokens, maximum chunk overlap, and more.

By default, we use OpenAI's `text-davinci-003` model. But you may choose to customize
the underlying LLM being used.

Below we show a few examples of LLM customization. This includes
- changing the underlying LLM 
- changing the number of output tokens (for OpenAI, Cohere, or AI21)
- having more fine-grained control over all parameters for any LLM, from input size to chunk overlap


## Example: Changing the underlying LLM

An example snippet of customizing the LLM being used is shown below. 
In this example, we use `text-davinci-002` instead of `text-davinci-003`. Note that 
you may plug in any LLM shown on Langchain's 
[LLM](https://langchain.readthedocs.io/en/latest/modules/llms.html) page.

```python

from gpt_index import (
    GPTKeywordTableIndex, 
    SimpleDirectoryReader, 
    LLMPredictor,
)
from langchain import OpenAI

documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))

# build index
index = GPTKeywordTableIndex(llm_predictor=llm_predictor)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```


## Example: Changing the number of output tokens (for OpenAI, Cohere, AI21)

The number of output tokens is usually set to some low number by default (for instance,
with OpenAI the default is 256).

For OpenAI, Cohere, AI21, you just need to set the `max_tokens` parameter 
(or maxTokens for AI21). We will handle text chunking/calculations under the hood.


```python

from gpt_index import (
    GPTKeywordTableIndex, 
    SimpleDirectoryReader, 
    LLMPredictor,
)
from langchain import OpenAI

documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=512))

# build index
index = GPTKeywordTableIndex(llm_predictor=llm_predictor)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```

If you are using other LLM classes from langchain, please see below.


## Example: Fine-grained control over all parameters

To have fine-grained control over all parameters, you will need to define
a custom PromptHelper class.


```python

from gpt_index import (
    GPTKeywordTableIndex, 
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper
)
from langchain import OpenAI

documents = SimpleDirectoryReader('data').load_data()


# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_output))

# build index
index = GPTKeywordTableIndex(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```
