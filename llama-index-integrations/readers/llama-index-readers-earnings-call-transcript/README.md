# EARNING CALL TRANSCRIPTS LOADER

```bash
pip install llama-index-readers-earnings-call-transcript
```

This loader fetches the earning call transcripts of US based companies from the website [discountingcashflows.com](https://discountingcashflows.com/). It is not available for commercial purposes

Install the required dependencies

```
pip install -r requirements.txt
```

The Earning call transcripts takes in three arguments

- Year
- Ticker symbol
- Quarter name from the list ["Q1","Q2","Q3","Q4"]

## Usage

```python
from llama_index.readers.earnings_call_transcript import EarningsCallTranscript

loader = EarningsCallTranscript(2023, "AAPL", "Q3")
docs = loader.load_data()
```

The metadata of the transcripts are the following

- ticker
- quarter
- date_time
- speakers_list

## Examples

#### Llama Index

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.earnings_call_transcript import EarningsCallTranscript

loader = EarningsCallTranscript(2023, "AAPL", "Q3")
docs = loader.load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query(
    "What was discussed about Generative AI?",
)
print(response)
```

#### Langchain

```python
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from llama_index.readers.earnings_call_transcript import EarningsCallTranscript

loader = EarningsCallTranscript(2023, "AAPL", "Q3")
docs = loader.load_data()

tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for questions about investor transcripts calls for a company. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="conversational-react-description")
agent.run("What was discussed about Generative AI?")
```
