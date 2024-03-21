## IMDB MOVIE REVIEWS LOADER

```bash
pip install llama-index-readers-imdb-review
```

This loader fetches all the reviews of a movie or a TV-series from IMDB official site. This loader is working on Windows machine and it requires further debug on Linux. Fixes are on the way

Install the required dependencies

```
pip install -r requirements.txt
```

The IMDB downloader takes in two attributes

- movie_name_year: The name of the movie or series and year
- webdriver_engine: To use edge, google or gecko (mozilla) webdriver
- generate_csv: Whether to generate csv file
- multithreading: whether to use multithreading or not

## Usage

```python
from llama_index.readers.imdb_review import IMDBReviews

loader = IMDBReviews(
    movie_name_year="The Social Network 2010", webdriver_engine="edge"
)
docs = loader.load_data()
```

The metadata has the following information

- date of the review (date)
- title of the review (title)
- rating of the review (rating)
- link of the review (link)
- whether the review is spoiler or not (spoiler)
- number of people found the review helpful (found_helpful)
- total number of votes (total)

It will download the files inside the folder `movie_reviews` with the filename as the movie name

## EXAMPLES

This loader can be used with both Langchain and LlamaIndex.

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.core import VectorStoreIndex

from llama_index.readers.imdb_review import IMDBReviews

loader = IMDBReviewsloader(
    movie_name_year="The Social Network 2010",
    webdriver_engine="edge",
    generate_csv=False,
    multithreading=False,
)
docs = loader.load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query(
    "What did the movie say about Mark Zuckerberg?",
)
print(response)
```

### Langchain

```python
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits.pandas import (
    create_pandas_dataframe_agent,
)
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

from llama_index.readers.imdb_review import IMDBReviews

loader = IMDBReviewsloader(
    movie_name_year="The Social Network 2010",
    webdriver_engine="edge",
    generate_csv=False,
    multithreading=False,
)
docs = loader.load_data()
tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you want to answer questions about the movies and their reviews. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]
llm = ChatOpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="conversational-react-description")
agent.run("What did the movie say about Mark Zuckerberg?")
```
