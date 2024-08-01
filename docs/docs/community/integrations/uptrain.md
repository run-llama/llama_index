# Perform Evaluations on LlamaIndex with UpTrain

**Overview**: In this example, we will see how to use UpTrain with LlamaIndex. UpTrain ([github](https://github.com/uptrain-ai/uptrain) || [website](https://github.com/uptrain-ai/uptrain/) || [docs](https://docs.uptrain.ai/)) is an open-source platform to evaluate and improve GenAI applications. It provides grades for 20+ preconfigured checks (covering language, code, embedding use cases), performs root cause analysis on failure cases and gives insights on how to resolve them. More details on UpTrain's evaluations can be found [here](https://github.com/uptrain-ai/uptrain?tab=readme-ov-file#pre-built-evaluations-we-offer-).

**Problem**: As an increasing number of companies are graduating their LLM prototypes to production-ready applications, their RAG pipelines are also getting complex. Developers are utilising modules like QueryRewrite, Context ReRank, etc., to enhance the accuracy of their RAG systems.

With increasing complexity comes more points of failure.

1. Advanced Evals are needed to evaluate the quality of these newer modules and determine if they actually improve the system's accuracy.
2. A robust experimentation framework is needed to systematically test different modules and make data-driven decisions.

**Solution**: UpTrain helps to solve for both:

1. UpTrain provides a series of checks to evaluate the quality of generated response, retrieved-context as well as all the interim steps. The relevant checks are ContextRelevance, SubQueryCompleteness, ContextReranking, ContextConciseness, FactualAccuracy, ContextUtilization, ResponseCompleteness, ResponseConciseness, etc.
2. UpTrain also allows you to experiment with different embedding models as well as have an "evaluate_experiments" method to compare different RAG configurations.

# How to go about it?

There are two ways you can use UpTrain with LlamaIndex:

1. **Using the UpTrain Callback Handler**: This method allows you to seamlessly integrate UpTrain with LlamaIndex. You can simply add UpTrainCallbackHandler to your existing LlamaIndex pipeline and it will evaluate all components of your RAG pipeline. This is the recommended method as it is the easiest to use and provides you with dashboards and insights with minimal effort.

2. **Using UpTrain's EvalLlamaIndex**: This method allows you to use UpTrain to perform evaluations on the generated responses. You can use the EvalLlamaIndex object to generate responses for the queries and then perform evaluations on the responses. You can find a detailed tutorial on how to do this below. This method offers more flexibility and control over the evaluations, but requires more effort to set up and use.

# 1. Using the UpTrain Callback Handler <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/callbacks/UpTrainCallback.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The following three demonstrations explain how you can use UpTrain Callback Handler to evaluate different components of your RAG pipelines.

## 1. **RAG Query Engine Evaluations**:

The RAG query engine plays a crucial role in retrieving context and generating responses. To ensure its performance and response quality, we conduct the following evaluations:

- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the retrieved context has sufficient information to answer the user query or not.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM's response can be verified via the retrieved context.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information required to answer the user query comprehensively.

## 2. **Sub-Question Query Generation Evaluation**:

The SubQuestionQueryGeneration operator decomposes a question into sub-questions, generating responses for each using an RAG query engine. To measure it's accuracy, we use:

- **[Sub Query Completeness](https://docs.uptrain.ai/predefined-evaluations/query-quality/sub-query-completeness)**: Assures that the sub-questions accurately and comprehensively cover the original query.

## 3. **Re-Ranking Evaluations**:

Re-ranking involves reordering nodes based on relevance to the query and choosing the top nodes. Different evaluations are performed based on the number of nodes returned after re-ranking.

a. Same Number of Nodes

- **[Context Reranking](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-reranking)**: Checks if the order of re-ranked nodes is more relevant to the query than the original order.

b. Different Number of Nodes:

- **[Context Conciseness](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-conciseness)**: Examines whether the reduced number of nodes still provides all the required information.

These evaluations collectively ensure the robustness and effectiveness of the RAG query engine, SubQuestionQueryGeneration operator, and the re-ranking process in the LlamaIndex pipeline.

#### **Note:**

- We have performed evaluations using a basic RAG query engine; the same evaluations can be performed using the advanced RAG query engine as well.
- Same is true for Re-Ranking evaluations, we have performed evaluations using SentenceTransformerRerank, the same evaluations can be performed using other re-rankers as well.

## Install Dependencies and Import Libraries

Install notebook dependencies.

```bash
%pip install llama-index-readers-web
%pip install llama-index-callbacks-uptrain
%pip install -q html2text llama-index pandas tqdm uptrain torch sentence-transformers
```

Import libraries.

```python
from getpass import getpass

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.callbacks import CallbackManager
from llama_index.callbacks.uptrain.base import UpTrainCallbackHandler
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai import OpenAI

import os
```

## Setup

UpTrain provides you with:
1. Dashboards with advanced drill-down and filtering options
1. Insights and common topics among failing cases
1. Observability and real-time monitoring of production data
1. Regression testing via seamless integration with your CI/CD pipelines

You can choose between the following options for evaluating using UpTrain:
### 1. **UpTrain's Open-Source Software (OSS)**:
You can use the open-source evaluation service to evaluate your model. In this case, you will need to provide an OpenAI API key. You can get yours [here](https://platform.openai.com/account/api-keys).

In order to view your evaluations in the UpTrain dashboard, you will need to set it up by running the following commands in your terminal:

```bash
git clone https://github.com/uptrain-ai/uptrain
cd uptrain
bash run_uptrain.sh
```

This will start the UpTrain dashboard on your local machine. You can access it at `http://localhost:3000/dashboard`.

Parameters:
- key_type="openai"
- api_key="OPENAI_API_KEY"
- project_name="PROJECT_NAME"


### 2. **UpTrain Managed Service and Dashboards**:
Alternatively, you can use UpTrain's managed service to evaluate your model. You can create a free UpTrain account [here](https://uptrain.ai/) and get free trial credits. If you want more trial credits, [book a call with the maintainers of UpTrain here](https://calendly.com/uptrain-sourabh/30min).

The benefits of using the managed service are:
1. No need to set up the UpTrain dashboard on your local machine.
1. Access to many LLMs without needing their API keys.

Once you perform the evaluations, you can view them in the UpTrain dashboard at `https://dashboard.uptrain.ai/dashboard`

Parameters:
- key_type="uptrain"
- api_key="UPTRAIN_API_KEY"
- project_name="PROJECT_NAME"

**Note:** The `project_name` will be the project name under which the evaluations performed will be shown in the UpTrain dashboard.

## Create the UpTrain Callback Handler

```python
os.environ["OPENAI_API_KEY"] = getpass()

callback_handler = UpTrainCallbackHandler(
    key_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    project_name="uptrain_llamaindex",
)

Settings.callback_manager = CallbackManager([callback_handler])
```

## Load and Parse Documents

Load documents from Paul Graham's essay "What I Worked On".

```python
documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
    ]
)
```

Parse the document into nodes.

```python
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
```

# 1. RAG Query Engine Evaluation

UpTrain callback handler will automatically capture the query, context and response once generated and will run the following three evaluations _(Graded from 0 to 1)_ on the response:

- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the retrieved context has sufficient information to answer the user query or not.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM's response can be verified via the retrieved context.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information required to answer the user query comprehensively.

```python
index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine()

max_characters_per_line = 80
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in queries:
    response = query_engine.query(query)
```

    Question: What did Paul Graham do growing up?
    Response: Paul Graham wrote short stories and started programming on the IBM 1401 in 9th grade using an early version of Fortran. Later, he convinced his father to buy a TRS-80, where he wrote simple games, a program to predict rocket heights, and a word processor.

    Context Relevance Score: 0.0
    Factual Accuracy Score: 1.0
    Response Completeness Score: 1.0


    Question: When and how did Paul Graham's mother die?
    Response: Paul Graham's mother died when he was 18 years old, from a brain tumor.

    Context Relevance Score: 0.0
    Factual Accuracy Score: 0.0
    Response Completeness Score: 1.0


    Question: What, in Paul Graham's opinion, is the most distinctive thing about YC?
    Response: The most distinctive thing about Y Combinator, according to Paul Graham, is that instead of deciding for himself what to work on, the problems come to him. Every 6 months, a new batch of startups brings their problems, which then become the focus of YC's work.

    Context Relevance Score: 0.0
    Factual Accuracy Score: 0.5
    Response Completeness Score: 1.0


    Question: When and how did Paul Graham meet Jessica Livingston?
    Response: Paul Graham met Jessica Livingston at a big party at his house in October 2003.

    Context Relevance Score: 1.0
    Factual Accuracy Score: 0.5
    Response Completeness Score: 1.0


    Question: What is Bel, and when and where was it written?
    Response: Bel is a new Lisp that was written in Arc. It was developed over a period of 4 years, from March 26, 2015 to October 12, 2019. Most of the work on Bel was done in England, where the author had moved to in the summer of 2016.

    Context Relevance Score: 1.0
    Factual Accuracy Score: 1.0
    Response Completeness Score: 1.0

Here's an example of the dashboard showing how you can filter and drill down to the failing cases and get insights on the failing cases:
![image-2.png](https://uptrain-assets.s3.ap-south-1.amazonaws.com/images/llamaindex/image-2.png)

# 2. Sub-Question Query Engine Evaluation

The **sub-question query engine** is used to tackle the problem of answering a complex query using multiple data sources. It first breaks down the complex query into sub-questions for each relevant data source, then gathers all the intermediate responses and synthesizes a final response.

UpTrain callback handler will automatically capture the sub-question and the responses for each of them once generated and will run the following three evaluations _(Graded from 0 to 1)_ on the response:

- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the retrieved context has sufficient information to answer the user query or not.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM's response can be verified via the retrieved context.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information required to answer the user query comprehensively.

In addition to the above evaluations, the callback handler will also run the following evaluation:

- **[Sub Query Completeness](https://docs.uptrain.ai/predefined-evaluations/query-quality/sub-query-completeness)**: Assures that the sub-questions accurately and comprehensively cover the original query.

```python
# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    documents=documents,
    use_async=True,
).as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="documents",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)
```

    Generated 3 sub questions.
    [1;3;38;2;237;90;200m[documents] Q: What did Paul Graham work on before Y Combinator?
    [0m[1;3;38;2;90;149;237m[documents] Q: What did Paul Graham work on during Y Combinator?
    [0m[1;3;38;2;11;159;203m[documents] Q: What did Paul Graham work on after Y Combinator?
    [0m[1;3;38;2;11;159;203m[documents] A: Paul Graham worked on a project with Robert and Trevor after Y Combinator.
    [0m[1;3;38;2;237;90;200m[documents] A: Paul Graham worked on projects with his colleagues Robert and Trevor before Y Combinator.
    [0m[1;3;38;2;90;149;237m[documents] A: Paul Graham worked on writing essays and working on Y Combinator during his time at Y Combinator.
    [0m


    Question: What did Paul Graham work on after Y Combinator?
    Response: Paul Graham worked on a project with Robert and Trevor after Y Combinator.

    Context Relevance Score: 0.0
    Factual Accuracy Score: 1.0
    Response Completeness Score: 0.5


    Question: What did Paul Graham work on before Y Combinator?
    Response: Paul Graham worked on projects with his colleagues Robert and Trevor before Y Combinator.

    Context Relevance Score: 0.0
    Factual Accuracy Score: 1.0
    Response Completeness Score: 0.5


    Question: What did Paul Graham work on during Y Combinator?
    Response: Paul Graham worked on writing essays and working on Y Combinator during his time at Y Combinator.

    Context Relevance Score: 0.0
    Factual Accuracy Score: 0.5
    Response Completeness Score: 0.5


    Question: How was Paul Grahams life different before, during, and after YC?
    Sub Query Completeness Score: 1.0

Here's an example of the dashboard visualizing the scores of the sub-questions in the form of a bar chart:

![image.png](https://uptrain-assets.s3.ap-south-1.amazonaws.com/images/llamaindex/image.png)

# 3. Re-ranking

Re-ranking is the process of reordering the nodes based on their relevance to the query. There are multiple classes of re-ranking algorithms offered by Llamaindex. We have used LLMRerank for this example.

The re-ranker allows you to enter the number of top n nodes that will be returned after re-ranking. If this value remains the same as the original number of nodes, the re-ranker will only re-rank the nodes and not change the number of nodes. Otherwise, it will re-rank the nodes and return the top n nodes.

We will perform different evaluations based on the number of nodes returned after re-ranking.

## 3a. Re-ranking (With same number of nodes)

If the number of nodes returned after re-ranking is the same as the original number of nodes, the following evaluation will be performed:

- **[Context Reranking](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-reranking)**: Checks if the order of re-ranked nodes is more relevant to the query than the original order.

```python
callback_handler = UpTrainCallbackHandler(
    key_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    project_name_prefix="llama",
)
Settings.callback_manager = CallbackManager([callback_handler])

rerank_postprocessor = SentenceTransformerRerank(
    top_n=3,  # number of nodes after reranking
    keep_retrieval_score=True,
)

index = VectorStoreIndex.from_documents(
    documents=documents,
)

query_engine = index.as_query_engine(
    similarity_top_k=3,  # number of nodes before reranking
    node_postprocessors=[rerank_postprocessor],
)

response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
```

    Question: What did Sam Altman do in this essay?
    Context Reranking Score: 0.0


    Question: What did Sam Altman do in this essay?
    Response: Sam Altman was asked to become the president of Y Combinator after the original founders decided to step back and reorganize the company for long-term sustainability.

    Context Relevance Score: 1.0
    Factual Accuracy Score: 1.0
    Response Completeness Score: 0.5

# 3b. Re-ranking (With different number of nodes)

If the number of nodes returned after re-ranking is the lesser as the original number of nodes, the following evaluation will be performed:

- **[Context Conciseness](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-conciseness)**: Examines whether the reduced number of nodes still provides all the required information.

```python
callback_handler = UpTrainCallbackHandler(
    key_type="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    project_name_prefix="llama",
)
Settings.callback_manager = CallbackManager([callback_handler])

rerank_postprocessor = SentenceTransformerRerank(
    top_n=2,  # Number of nodes after re-ranking
    keep_retrieval_score=True,
)

index = VectorStoreIndex.from_documents(
    documents=documents,
)
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Number of nodes before re-ranking
    node_postprocessors=[rerank_postprocessor],
)

# Use your advanced RAG
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
```

    Question: What did Sam Altman do in this essay?
    Context Conciseness Score: 0.0


    Question: What did Sam Altman do in this essay?
    Response: Sam Altman offered unsolicited advice to the author during a visit to California for interviews.


    Context Relevance Score: 1.0
    Factual Accuracy Score: 1.0
    Response Completeness Score: 0.5

# UpTrain's Managed Service Dashboard and Insights

To use the UpTrain's managed service via the UpTrain callback handler, the only change required is to set the `key_type` and `api_key` parameters. The rest of the code remains the same.

```python
callback_handler = UpTrainCallbackHandler(
    key_type="uptrain",
    api_key="up-******************************",
    project_name_prefix="llama",
)
```

Here's a short GIF showcasing the dashboard and the insights that you can get from the UpTrain managed service:

![output.gif](https://uptrain-assets.s3.ap-south-1.amazonaws.com/images/llamaindex/output.gif)

# 2. Using UpTrain's EvalLlamaIndex <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/UpTrain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Install UpTrain and LlamaIndex

```bash
pip install uptrain llama_index
```

## Import required libraries

```python
import httpx
import os
import openai
import pandas as pd

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from uptrain import Evals, EvalLlamaIndex, Settings as UpTrainSettings
```

## Create the dataset folder for the query engine

You can use any documents that you have to do this. For this tutorial, we will use data on New York City extracted from wikipedia. We will only add one document to the folder, but you can add as many as you want.

```python
url = "https://uptrain-assets.s3.ap-south-1.amazonaws.com/data/nyc_text.txt"
if not os.path.exists("nyc_wikipedia"):
    os.makedirs("nyc_wikipedia")
dataset_path = os.path.join("./nyc_wikipedia", "nyc_text.txt")

if not os.path.exists(dataset_path):
    r = httpx.get(url)
    with open(dataset_path, "wb") as f:
        f.write(r.content)
```

## Make the list of queries

Before we can generate responses, we need to create a list of queries. Since the query engine is trained on New York City, we will create a list of queries related to New York City.

```python
data = [
    {"question": "What is the population of New York City?"},
    {"question": "What is the area of New York City?"},
    {"question": "What is the largest borough in New York City?"},
    {"question": "What is the average temperature in New York City?"},
    {"question": "What is the main airport in New York City?"},
    {"question": "What is the famous landmark in New York City?"},
    {"question": "What is the official language of New York City?"},
    {"question": "What is the currency used in New York City?"},
    {"question": "What is the time zone of New York City?"},
    {"question": "What is the famous sports team in New York City?"},
]
```

**This notebook uses the OpenAI API to generate text for prompts as well as to create the Vector Store Index. So, set openai.api_key to your OpenAI API key.**

```python
openai.api_key = "sk-************************"  # your OpenAI API key
```

## Create a query engine using LlamaIndex

Let's create a vector store index using LLamaIndex and then use that as a query engine to retrieve relevant sections from the documentation.

```python
Settings.chunk_size = 512

documents = SimpleDirectoryReader("./nyc_wikipedia/").load_data()

vector_index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = vector_index.as_query_engine()
```

## Setup

UpTrain provides you with:
1. Dashboards with advanced drill-down and filtering options
1. Insights and common topics among failing cases
1. Observability and real-time monitoring of production data
1. Regression testing via seamless integration with your CI/CD pipelines

You can choose between the following two alternatives for evaluating using UpTrain:

# Alternative 1: Evaluate using UpTrain's Open-Source Software (OSS)

You can use the open-source evaluation service to evaluate your model. In this case, you will need to provide an OpenAI API key. You can get yours [here](https://platform.openai.com/account/api-keys).

In order to view your evaluations in the UpTrain dashboard, you will need to set it up by running the following commands in your terminal:

```bash
git clone https://github.com/uptrain-ai/uptrain
cd uptrain
bash run_uptrain.sh
```

This will start the UpTrain dashboard on your local machine. You can access it at `http://localhost:3000/dashboard`.

**Note:** The `project_name` will be the project name under which the evaluations performed will be shown in the UpTrain dashboard.

```python
settings = UpTrainSettings(
    openai_api_key=openai.api_key,
)
```

## Create the EvalLlamaIndex object

Now that we have created the query engine, we can use it to create an EvalLlamaIndex object. This object will be used to generate responses for the queries.

```python
llamaindex_object = EvalLlamaIndex(
    settings=settings, query_engine=query_engine
)
```

## Run the evaluation

Now that we have the list of queries, we can use the EvalLlamaIndex object to generate responses for the queries and then perform evaluations on the responses. You can find an extensive list of the evaluations offered by UpTrain [here](https://docs.uptrain.ai/key-components/evals). We have chosen two that we found to be the most relevant for this tutorial:

1. **Context Relevance**: This evaluation checks whether the retrieved context is relevant to the query. This is important because the retrieved context is used to generate the response. If the retrieved context is not relevant to the query, then the response will not be relevant to the query either.

2. **Response Conciseness**: This evaluation checks whether the response is concise. This is important because the response should be concise and should not contain any unnecessary information.

```python
results = llamaindex_object.evaluate(
    project_name="uptrain-llama-index",
    evaluation_name="nyc_wikipedia",  # adding project and evaluation names allow you to track the results in the UpTrain dashboard
    data=data,
    checks=[Evals.CONTEXT_RELEVANCE, Evals.RESPONSE_CONCISENESS],
)
```

```python
pd.DataFrame(results)
```

# Alternative 2: Evaluate using UpTrain's Managed Service and Dashboards

Alternatively, you can use UpTrain's managed service to evaluate your model. You can create a free UpTrain account [here](https://uptrain.ai/) and get free trial credits. If you want more trial credits, [book a call with the maintainers of UpTrain here](https://calendly.com/uptrain-sourabh/30min).

The benefits of using the managed service are:
1. No need to set up the UpTrain dashboard on your local machine.
1. Access to many LLMs without needing their API keys.

Once you perform the evaluations, you can view them in the UpTrain dashboard at `https://dashboard.uptrain.ai/dashboard`

**Note:** The `project_name` will be the project name under which the evaluations performed will be shown in the UpTrain dashboard.

```python
UPTRAIN_API_KEY = "up-**********************"  # your UpTrain API key

# We use `uptrain_access_token` parameter instead of 'openai_api_key' in settings in this case
settings = UpTrainSettings(
    uptrain_access_token=UPTRAIN_API_KEY,
)
```

## Create the EvalLlamaIndex object

Now that we have created the query engine, we can use it to create an EvalLlamaIndex object. This object will be used to generate responses for the queries.

```python
llamaindex_object = EvalLlamaIndex(
    settings=settings, query_engine=query_engine
)
```

## Run the evaluation

Now that we have the list of queries, we can use the EvalLlamaIndex object to generate responses for the queries and then perform evaluations on the responses. You can find an extensive list of the evaluations offered by UpTrain [here](https://docs.uptrain.ai/key-components/evals). We have chosen two that we found to be the most relevant for this tutorial:

1. **Context Relevance**: This evaluation checks whether the retrieved context is relevant to the query. This is important because the retrieved context is used to generate the response. If the retrieved context is not relevant to the query, then the response will not be relevant to the query either.

2. **Response Conciseness**: This evaluation checks whether the response is concise. This is important because the response should be concise and should not contain any unnecessary information.

```python
results = llamaindex_object.evaluate(
    project_name="uptrain-llama-index",
    evaluation_name="nyc_wikipedia",  # adding project and evaluation names allow you to track the results in the UpTrain dashboard
    data=data,
    checks=[Evals.CONTEXT_RELEVANCE, Evals.RESPONSE_CONCISENESS],
)
```

```python
pd.DataFrame(results)
```

### Dashboards:

Histogram of score vs number of cases with that score

![nyc_dashboard.png](https://uptrain-assets.s3.ap-south-1.amazonaws.com/images/llamaindex/nyc_dashboard.png)

You can filter failure cases and generate common topics among them. This can help identify the core issue and help fix it

![nyc_insights.png](https://uptrain-assets.s3.ap-south-1.amazonaws.com/images/llamaindex/nyc_insights.png)

## Learn More

1. [Colab Notebook on UpTrainCallbackHandler](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/callbacks/UpTrainCallback.ipynb)
1. [Colab Notebook on UpTrain Integration with LlamaIndex](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/UpTrain.ipynb)
1. [UpTrain Github Repository](https://github.com/uptrain-ai/uptrain)
1. [UpTrain Documentation](https://docs.uptrain.ai/)
