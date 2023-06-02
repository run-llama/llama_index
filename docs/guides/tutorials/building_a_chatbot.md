# 💬🤖 How to Build a Chatbot

LlamaIndex is an interface between your data and LLM's; it offers the toolkit for you to setup a query interface around your data for any downstream task, whether it's question-answering, summarization, or more.

In this tutorial, we show you how to build a context augmented chatbot. We use Langchain for the underlying Agent/Chatbot abstractions, and we use LlamaIndex for the data retrieval/lookup/querying! The result is a chatbot agent that has access to a rich set of "data interface" Tools that LlamaIndex provides to answer queries over your data.

**Note**: This is a continuation of some initial work building a query interface over SEC 10-K filings - [check it out here](https://medium.com/@jerryjliu98/how-unstructured-and-llamaindex-can-help-bring-the-power-of-llms-to-your-own-data-3657d063e30d).

### Context

In this tutorial, we build an "10-K Chatbot" by downloading the raw UBER 10-K HTML filings from Dropbox. The user can choose to ask questions regarding the 10-K filings.

### Ingest Data

Let's first download the raw 10-k files, from 2019-2022.

```python
# NOTE: the code examples assume you're operating within a Jupyter notebook.
# download files
!mkdir data
!wget "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1" -O data/UBER.zip
!unzip data/UBER.zip -d data

```

We use the [Unstructured](https://github.com/Unstructured-IO/unstructured) library to parse the HTML files into formatted text.
We have a direct integration with Unstructured through [LlamaHub](https://llamahub.ai/) - this allows us to convert any text into a Document format that LlamaIndex can ingest.

```python

from llama_index import download_loader, GPTVectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from pathlib import Path

years = [2022, 2021, 2020, 2019]
UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(file=Path(f'./data/UBER/UBER_{year}.html'), split_documents=False)
    # insert year metadata into each year
    for d in year_docs:
        d.extra_info = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)
```

### Setting up Vector Indices for each year

We first setup a vector index for each year. Each vector index allows us 
to ask questions about the 10-K filing of a given year.

We build each index and save it to disk.

```python
# initialize simple vector indices + global vector index
service_context = ServiceContext.from_defaults(chunk_size=512)
index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = GPTVectorStoreIndex.from_documents(
        doc_set[year], 
        service_context=service_context,
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f'./storage/{year}')

```

To load an index from disk, do the following
```python
# Load indices from disk
index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(persist_dir=f'./storage/{year}')
    cur_index = load_index_from_storage(storage_context=storage_context)
    index_set[year] = cur_index
```


### Composing a Graph to Synthesize Answers Across 10-K Filings

Since we have access to documents of 4 years, we may not only want to ask questions regarding the 10-K document of a given year, but ask questions that require analysis over all 10-K filings. 

To address this, we compose a "graph" which consists of a list index defined over the 4 vector indices. Querying this graph would first retrieve information from each vector index, and combine information together via the list index.

```python
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, load_graph_from_storage
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph

# describe each index to help traversal of composed graph
index_summaries = [f"UBER 10-k Filing for {year} fiscal year" for year in years]

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults()

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[y] for y in years], 
    index_summaries=index_summaries,
    service_context=service_context,
    storage_context = storage_context,
)
root_id = graph.root_id

# [optional] save to disk
storage_context.persist(persist_dir=f'./storage/root')

# [optional] load from disk, so you don't need to build graph from scratch
graph = load_graph_from_storage(
    root_id=root_id, 
    service_context=service_context,
    storage_context=storage_context,
)

```

### Setting up the Tools + Langchain Chatbot Agent

We use Langchain to setup the outer chatbot agent, which has access to a set of Tools.
LlamaIndex provides some wrappers around indices and graphs so that they can be easily used within a Tool interface.

```python
# do imports
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
```

We want to define a separate Tool for each index (corresponding to a given year), as well 
as the graph. We can define all tools under a central `LlamaToolkit` interface.

Below, we define a `IndexToolConfig` for our graph. Note that we also import a `DecomposeQueryTransform` module for use within each vector index within the graph - this allows us to "decompose" the overall query into a query that can be answered from each subindex. (see example below).

```python
# define a decompose transform
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define custom retrievers
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={'index_summary': index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode='tree_summarize',
    verbose=True,
)

# tool config
graph_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
    tool_kwargs={"return_direct": True}
)
```

Besides the `GraphToolConfig` object, we also define an `IndexToolConfig` corresponding to each index:

```python
# define toolkit
index_configs = []
for y in range(2019, 2023):
    query_engine = index_set[y].as_query_engine(
        similarity_top_k=3,
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine, 
        name=f"Vector Index {y}",
        description=f"useful for when you want to answer queries about the {y} SEC 10-K for Uber",
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)
```

Finally, we combine these configs with our `LlamaToolkit`: 

```python
toolkit = LlamaToolkit(
    index_configs=index_configs + [graph_config],
)
```


Finally, we call `create_llama_chat_agent` to create our Langchain chatbot agent, which
has access to the 5 Tools we defined above:

```python
memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)
```

### Testing the Agent

We can now test the agent with various queries.

If we test it with a simple "hello" query, the agent does not use any Tools.

```python
agent_chain.run(input="hi, i am bob")
```

```
> Entering new AgentExecutor chain...

Thought: Do I need to use a tool? No
AI: Hi Bob, nice to meet you! How can I help you today?

> Finished chain.
'Hi Bob, nice to meet you! How can I help you today?'
```

If we test it with a query regarding the 10-k of a given year, the agent will use
the relevant vector index Tool.

```python
agent_chain.run(input="What were some of the biggest risk factors in 2020 for Uber?")
```

```
> Entering new AgentExecutor chain...

Thought: Do I need to use a tool? Yes
Action: Vector Index 2020
Action Input: Risk Factors
...

Observation: 

Risk Factors

The COVID-19 pandemic and the impact of actions to mitigate the pandemic has adversely affected and continues to adversely affect our business, financial condition, and results of operations.

...
'\n\nRisk Factors\n\nThe COVID-19 pandemic and the impact of actions to mitigate the pandemic has adversely affected and continues to adversely affect our business,

```

Finally, if we test it with a query to compare/contrast risk factors across years,
the agent will use the graph index Tool.

```python
cross_query_str = (
    "Compare/contrast the risk factors described in the Uber 10-K across years. Give answer in bullet points."
)
agent_chain.run(input=cross_query_str)
```

```
> Entering new AgentExecutor chain...

Thought: Do I need to use a tool? Yes
Action: Graph Index
Action Input: Compare/contrast the risk factors described in the Uber 10-K across years.> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2022 fiscal year?
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2022 fiscal year?
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 964 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 18 tokens
> Got response: 
The risk factors described in the Uber 10-K for the 2022 fiscal year include: the potential for changes in the classification of Drivers, the potential for increased competition, the potential for...
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2021 fiscal year?
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2021 fiscal year?
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 590 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 18 tokens
> Got response: 
1. The COVID-19 pandemic and the impact of actions to mitigate the pandemic have adversely affected and may continue to adversely affect parts of our business.

2. Our business would be adversely ...
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2020 fiscal year?
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2020 fiscal year?
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 516 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 18 tokens
> Got response: 
The risk factors described in the Uber 10-K for the 2020 fiscal year include: the timing of widespread adoption of vaccines against the virus, additional actions that may be taken by governmental ...
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2019 fiscal year?
> Current query: Compare/contrast the risk factors described in the Uber 10-K across years.
> New query:  What are the risk factors described in the Uber 10-K for the 2019 fiscal year?
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 1020 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 18 tokens
INFO:llama_index.indices.common.tree.base:> Building index from nodes: 0 chunks
> Got response: 
Risk factors described in the Uber 10-K for the 2019 fiscal year include: competition from other transportation providers; the impact of government regulations; the impact of litigation; the impac...
INFO:llama_index.token_counter.token_counter:> [query] Total LLM token usage: 7039 tokens
INFO:llama_index.token_counter.token_counter:> [query] Total embedding token usage: 72 tokens

Observation: 
In 2020, the risk factors included the timing of widespread adoption of vaccines against the virus, additional actions that may be taken by governmental authorities, the further impact on the business of Drivers

...

```


### Setting up the Chatbot Loop

Now that we have the chatbot setup, it only takes a few more steps to setup a basic interactive loop to converse with our SEC-augmented chatbot! 

```python
while True:
    text_input = input("User: ")
    response = agent_chain.run(input=text_input)
    print(f'Agent: {response}')
    
```

Here's an example of the loop in action:
```
User:  What were some of the legal proceedings against Uber in 2022?
Agent: 

In 2022, legal proceedings against Uber include a motion to compel arbitration, an appeal of a ruling that Proposition 22 is unconstitutional, a complaint alleging that drivers are employees and entitled to protections under the wage and labor laws, a summary judgment motion, allegations of misclassification of drivers and related employment violations in New York, fraud related to certain deductions, class actions in Australia alleging that Uber entities conspired to injure the group members during the period 2014 to 2017 by either directly breaching transport legislation or commissioning offenses against transport legislation by UberX Drivers in Australia, and claims of lost income and decreased value of certain taxi. Additionally, Uber is facing a challenge in California Superior Court alleging that Proposition 22 is unconstitutional, and a preliminary injunction order prohibiting Uber from classifying Drivers as independent contractors and from violating various wage and hour laws.

User: 

```

### Notebook

Take a look at our [corresponding notebook](https://github.com/jerryjliu/llama_index/blob/main/examples/chatbot/Chatbot_SEC.ipynb). 
