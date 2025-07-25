# How to Build a Chatbot

LlamaIndex serves as a bridge between your data and Large Language Models (LLMs), providing a toolkit that enables you to establish a query interface around your data for a variety of tasks, such as question-answering and summarization.

In this tutorial, we'll walk you through building a context-augmented chatbot using a [Data Agent](https://gpt-index.readthedocs.io/en/stable/core_modules/agent_modules/agents/root.html). This agent, powered by LLMs, is capable of intelligently executing tasks over your data. The end result is a chatbot agent equipped with a robust set of data interface tools provided by LlamaIndex to answer queries about your data.

**Note**: This tutorial builds upon initial work on creating a query interface over SEC 10-K filings - [check it out here](https://medium.com/@jerryjliu98/how-unstructured-and-llamaindex-can-help-bring-the-power-of-llms-to-your-own-data-3657d063e30d).

### Context

In this guide, we’ll build a "10-K Chatbot" that uses raw UBER 10-K HTML filings from Dropbox. Users can interact with the chatbot to ask questions related to the 10-K filings.

### Preparation

```python
import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio

nest_asyncio.apply()
```

### Ingest Data

Let's first download the raw 10-k files, from 2019-2022.

```
# NOTE: the code examples assume you're operating within a Jupyter notebook.
# download files
!mkdir data
!wget "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1" -O data/UBER.zip
!unzip data/UBER.zip -d data
```

To parse the HTML files into formatted text, we use the [Unstructured](https://github.com/Unstructured-IO/unstructured) library. Thanks to [LlamaHub](https://llamahub.ai/), we can directly integrate with Unstructured, allowing conversion of any text into a Document format that LlamaIndex can ingest.

First we install the necessary packages:

```
!pip install llama-hub unstructured
```

Then we can use the `UnstructuredReader` to parse the HTML files into a list of `Document` objects.

```python
from llama_index.readers.file import UnstructuredReader
from pathlib import Path

years = [2022, 2021, 2020, 2019]

loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
    )
    # insert year metadata into each year
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)
```

### Setting up Vector Indices for each year

We first setup a vector index for each year. Each vector index allows us
to ask questions about the 10-K filing of a given year.

We build each index and save it to disk.

```python
# initialize simple vector indices
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings

Settings.chunk_size = 512
index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"./storage/{year}")
```

To load an index from disk, do the following

```python
# Load indices from disk
from llama_index.core import load_index_from_storage

index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"./storage/{year}"
    )
    cur_index = load_index_from_storage(
        storage_context,
    )
    index_set[year] = cur_index
```

### Setting up a Sub Question Query Engine to Synthesize Answers Across 10-K Filings

Since we have access to documents of 4 years, we may not only want to ask questions regarding the 10-K document of a given year, but ask questions that require analysis over all 10-K filings.

To address this, we can use a [Sub Question Query Engine](https://gpt-index.readthedocs.io/en/stable/examples/query_engine/sub_question_query_engine.html). It decomposes a query into subqueries, each answered by an individual vector index, and synthesizes the results to answer the overall query.

LlamaIndex provides some wrappers around indices (and query engines) so that they can be used by query engines and agents. First we define a `QueryEngineTool` for each vector index.
Each tool has a name and a description; these are what the LLM agent sees to decide which tool to choose.

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
        ),
    )
    for year in years
]
```

Now we can create the Sub Question Query Engine, which will allow us to synthesize answers across the 10-K filings. We pass in the `individual_query_engine_tools` we defined above, as well as an `llm` that will be used to run the subqueries.

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(model="gpt-3.5-turbo"),
)
```

### Setting up the Chatbot Agent

We use a LlamaIndex Data Agent to setup the outer chatbot agent, which has access to a set of Tools. Specifically, we will use a `FunctionAgent`, that takes advantage of OpenAI API function calling. We want to use the separate Tools we defined previously for each index (corresponding to a given year), as well as a tool for the sub question query engine we defined above.

First we define a `QueryEngineTool` for the sub question query engine:

```python
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
    ),
)
```

Then, we combine the Tools we defined above into a single list of tools for the agent:

```python
tools = individual_query_engine_tools + [query_engine_tool]
```

Finally, we call `FunctionAgent()` to create the agent, passing in the list of tools we defined above.

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4.1"))
```

### Testing the Agent

We can now test the agent with various queries.

If we test it with a simple "hello" query, the agent does not use any Tools.

```python
response = await agent.run("hi, i am bob")
print(str(response))
```

```
Hello Bob! How can I assist you today?
```

If we test it with a query regarding the 10-k of a given year, the agent will use
the relevant vector index Tool.

```python
response = await agent.run(
    "What were some of the biggest risk factors in 2020 for Uber?"
)
print(str(response))
```

```
Some of the biggest risk factors for Uber in 2020 were:

1. The adverse impact of the COVID-19 pandemic and actions taken to mitigate it on the business.
2. The potential reclassification of drivers as employees, workers, or quasi-employees instead of independent contractors.
3. Intense competition in the mobility, delivery, and logistics industries, with low-cost alternatives and well-capitalized competitors.
4. The need to lower fares or service fees and offer driver incentives and consumer discounts to remain competitive.
5. Significant losses incurred and the uncertainty of achieving profitability.
6. The risk of not attracting or maintaining a critical mass of platform users.
7. Operational, compliance, and cultural challenges related to the workplace culture and forward-leaning approach.
8. The potential negative impact of international investments and the challenges of conducting business in foreign countries.
9. Risks associated with operational and compliance challenges, localization, laws and regulations, competition, social acceptance, technological compatibility, improper business practices, liability uncertainty, managing international operations, currency fluctuations, cash transactions, tax consequences, and payment fraud.

These risk factors highlight the challenges and uncertainties that Uber faced in 2020.
```

Finally, if we test it with a query to compare/contrast risk factors across years,
the agent will use the Sub Question Query Engine Tool.

```python
cross_query_str = "Compare/contrast the risk factors described in the Uber 10-K across years. Give answer in bullet points."

response = await agent.run(cross_query_str)
print(str(response))
```

```
Here is a comparison of the risk factors described in the Uber 10-K reports across years:

2022 Risk Factors:
- Potential adverse effect if drivers were classified as employees instead of independent contractors.
- Highly competitive nature of the mobility, delivery, and logistics industries.
- Need to lower fares or service fees to remain competitive.
- History of significant losses and expectation of increased operating expenses.
- Impact of future pandemics or disease outbreaks on the business and financial results.
- Potential harm to the business due to economic conditions and their effect on discretionary consumer spending.

2021 Risk Factors:
- Adverse impact of the COVID-19 pandemic and actions to mitigate it on the business.
- Potential reclassification of drivers as employees instead of independent contractors.
- Highly competitive nature of the mobility, delivery, and logistics industries.
- Need to lower fares or service fees and offer incentives to remain competitive.
- History of significant losses and uncertainty of achieving profitability.
- Importance of attracting and maintaining a critical mass of platform users.

2020 Risk Factors:
- Adverse impact of the COVID-19 pandemic on the business.
- Potential reclassification of drivers as employees.
- Highly competitive nature of the mobility, delivery, and logistics industries.
- Need to lower fares or service fees to remain competitive.
- History of significant losses and potential future expenses.
- Importance of attracting and maintaining a critical mass of platform users.
- Operational and cultural challenges faced by the company.

2019 Risk Factors:
- Competition with local companies.
- Differing levels of social acceptance.
- Technological compatibility issues.
- Exposure to improper business practices.
- Legal uncertainty.
- Difficulties in managing international operations.
- Fluctuations in currency exchange rates.
- Regulations governing local currencies.
- Tax consequences.
- Financial accounting burdens.
- Difficulties in implementing financial systems.
- Import and export restrictions.
- Political and economic instability.
- Public health concerns.
- Reduced protection for intellectual property rights.
- Limited influence over minority-owned affiliates.
- Regulatory complexities.

These comparisons highlight both common and unique risk factors that Uber faced in different years.
```

### Setting up the Chatbot Loop

Now that we have the chatbot setup, it only takes a few more steps to setup a basic interactive loop to chat with our SEC-augmented chatbot!

```python
agent = FunctionAgent(tools=tools, llm=OpenAI(model="gpt-4.1"))

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = await agent.run(text_input)
    print(f"Agent: {response}")
```

Here's an example of the loop in action:

```
User:  What were some of the legal proceedings against Uber in 2022?
Agent: In 2022, Uber faced several legal proceedings. Some of the notable ones include:

1. Petition against Proposition 22: A petition was filed in California alleging that Proposition 22, which classifies app-based drivers as independent contractors, is unconstitutional.

2. Lawsuit by Massachusetts Attorney General: The Massachusetts Attorney General filed a lawsuit against Uber, claiming that drivers should be classified as employees and entitled to protections under wage and labor laws.

3. Allegations by New York Attorney General: The New York Attorney General made allegations against Uber regarding the misclassification of drivers and related employment violations.

4. Swiss social security rulings: Swiss social security rulings classified Uber drivers as employees, which could have implications for Uber's operations in Switzerland.

5. Class action lawsuits in Australia: Uber faced class action lawsuits in Australia, with allegations that the company conspired to harm participants in the taxi, hire-car, and limousine industries.

It's important to note that the outcomes of these legal proceedings are uncertain and may vary.

User:

```

### Notebook

Take a look at our [corresponding notebook](../../../examples/agent/Chatbot_SEC.ipynb).
