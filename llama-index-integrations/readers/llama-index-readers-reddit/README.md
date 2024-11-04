# Reddit Reader

```bash
pip install llama-index-readers-reddit
```

For any subreddit(s) you're interested in, search for relevant posts using keyword(s) and load the resulting text in the post and and top-level comments into LLMs/ LangChains.

## Get your Reddit credentials ready

1. Visit Reddit App Preferences (https://www.reddit.com/prefs/apps) or [https://old.reddit.com/prefs/apps/](https://old.reddit.com/prefs/apps/)
2. Scroll to the bottom and click "create another app..."
3. Fill out the name, description, and redirect url for your app, then click "create app"
4. Now you should be able to see the personal use script, secret, and name of your app. Store those as environment variables REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT respectively.
5. Additionally store the environment variables REDDIT_USERNAME and REDDIT_PASSWORD, which correspond to the credentials for your Reddit account.

## Usage

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, download_loader

from llama_index.readers.reddit import RedditReader

subreddits = ["MachineLearning"]
search_keys = ["PyTorch", "deploy"]
post_limit = 10

loader = RedditReader()
documents = loader.load_data(
    subreddits=subreddits, search_keys=search_keys, post_limit=post_limit
)
index = VectorStoreIndex.from_documents(documents)

index.query("What are the pain points of PyTorch users?")
```

### LangChain

```python
from llama_index.core import VectorStoreIndex, download_loader

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

from llama_index.readers.reddit import RedditReader

subreddits = ["MachineLearning"]
search_keys = ["PyTorch", "deploy"]
post_limit = 10

loader = RedditReader()
documents = loader.load_data(
    subreddits=subreddits, search_keys=search_keys, post_limit=post_limit
)
index = VectorStoreIndex.from_documents(documents)

tools = [
    Tool(
        name="Reddit Index",
        func=lambda q: index.query(q),
        description=f"Useful when you want to read relevant posts and top-level comments in subreddits.",
    ),
]
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)

output = agent_chain.run(input="What are the pain points of PyTorch users?")
print(output)
```

This loader is designed to be used as a way to load data into [GPT Index](https://github.com/run-llama/llama_index/).
