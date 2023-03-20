# Defining LLMs

The goal of LlamaIndex is to provide a toolkit of data structures that can organize external information in a manner that
is easily compatible with the prompt limitations of an LLM. Therefore LLMs are always used to construct the final
answer.
Depending on the [type of index](/reference/indices.rst) being used,
LLMs may also be used during index construction, insertion, and query traversal.

LlamaIndex uses Langchain's [LLM](https://langchain.readthedocs.io/en/latest/modules/llms.html)
and [LLMChain](https://langchain.readthedocs.io/en/latest/modules/chains.html) module to define
the underlying abstraction. We introduce a wrapper class,
[`LLMPredictor`](/reference/llm_predictor.rst), for integration into LlamaIndex.

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
In this example, we use `text-davinci-002` instead of `text-davinci-003`. Available models include `text-davinci-003`,`text-curie-001`,`text-babbage-001`,`text-ada-001`, `code-davinci-002`,`code-cushman-001`. Note that
you may plug in any LLM shown on Langchain's
[LLM](https://langchain.readthedocs.io/en/latest/modules/llms.html) page.

```python

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
)
from langchain import OpenAI

documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))

# build index
index = GPTKeywordTableIndex(documents, llm_predictor=llm_predictor)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```

## Example: Changing the number of output tokens (for OpenAI, Cohere, AI21)

The number of output tokens is usually set to some low number by default (for instance,
with OpenAI the default is 256).

For OpenAI, Cohere, AI21, you just need to set the `max_tokens` parameter
(or maxTokens for AI21). We will handle text chunking/calculations under the hood.

```python

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
)
from langchain import OpenAI

documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=512))

# build index
index = GPTKeywordTableIndex(documents, llm_predictor=llm_predictor)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```

If you are using other LLM classes from langchain, please see below.

## Example: Fine-grained control over all parameters

To have fine-grained control over all parameters, you will need to define
a custom PromptHelper class.

```python

from llama_index import (
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
index = GPTKeywordTableIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# get response from query
response = index.query("What did the author do after his time at Y Combinator?")

```

## Example: Using a Custom LLM Model

To use a custom LLM model, you only need to implement the `LLM` class [from Langchain](https://langchain.readthedocs.io/en/latest/modules/llms/examples/custom_llm.html). You will be responsible for passing the text to the model and returning the newly generated tokens.

Here is a small example using locally running FLAN-T5 model and Huggingface's pipeline abstraction:

```python
import torch
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex
from llama_index import LLMPredictor
from langchain.llms.base import LLM
from transformers import pipeline


class FlanLLM(LLM):
    model_name = "google/flan-t5-xl"
    pipeline = pipeline("text2text-generation", model=model_name, device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=512)[0]["generated_text"]

    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


llm_predictor = LLMPredictor(llm=FlanLLM())


documents = SimpleDirectoryReader('data').load_data()
index = GPTListIndex(documents, llm_predictor=llm_predictor)

index.save_to_disk('index.json')

new_index = GPTListIndex.load_from_disk('index.json', embed_model=embed_model)
new_index.query("<query_text>")
```

Using this method, you can use any LLM. Maybe you have one running locally, or running on your own server. As long as the class is implemented and the generated tokens are returned, it should work out.

Note that you may have to adjust the internal prompts to get good performance. Even then, you should be using a sufficiently large LLM to ensure it's capable of handling the complex queries that LlamaIndex uses internally, so your mileage may vary.

A list of all default internal prompts is available [here](https://github.com/jerryjliu/llama_index/blob/main/gpt_index/prompts/default_prompts.py), and chat-specific prompts are listed [here](https://github.com/jerryjliu/llama_index/blob/main/gpt_index/prompts/chat_prompts.py). You can also implement your own custom prompts, as described [here](https://gpt-index.readthedocs.io/en/latest/how_to/custom_prompts.html).
