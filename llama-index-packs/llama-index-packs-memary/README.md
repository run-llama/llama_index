# memary Pack

Agents use LLMs that are currently constrained to finite context windows. memary overcomes this limitation by allowing your agents to store a large corpus of information in knowledge graphs, infer user knowledge through our memory modules, and only retrieve relevant information for meaningful responses.

## CLI Usage

You can download llamapacks directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamapack memary --download-dir ./memary
```

You can then inspect the files at `./memary` and use them as a template for your own project.

## Demo

**Notes:** memary currently assumes the local installation method and currently supports any models available through Ollama:

- LLM running locally using Ollama (Llama 3 8B/40B as suggested defaults) **OR** `gpt-3.5-turbo`
- Vision model running locally using Ollama (LLaVA as suggested default) **OR** `gpt-4-vision-preview`

memary will default to the locally run models unless explicitly specified.

**To run the Streamlit app:**

1. [Optional] If running models locally using Ollama, follow this the instructions in this [repo](https://github.com/ollama/ollama).

2. Ensure that a `.env` exists with any necessary API keys and Neo4j credentials.

```
OPENAI_API_KEY="YOUR_API_KEY"
NEO4J_PW="YOUR_NEO4J_PW"
NEO4J_URL="YOUR_NEO4J_URL"
PERPLEXITY_API_KEY="YOUR_API_KEY"
GOOGLEMAPS_API_KEY="YOUR_API_KEY"
ALPHA_VANTAGE_API_KEY="YOUR_API_KEY"
```

3. How to get API keys:

```
OpenAI key: https://openai.com/index/openai-api

Neo4j: https://neo4j.com/cloud/platform/aura-graph-database/?ref=nav-get-started-cta
   Click 'Start for free'
   Create a free instance
   Open auto-downloaded txt file and use the credentials

Perplexity key: https://www.perplexity.ai/settings/api

Google Maps:
   Keys are generated in the 'Credentials' page of the 'APIs & Services' tab of Google Cloud Console https://console.cloud.google.com/apis/credentials

Alpha Vantage: (this key is for getting real time stock data)
  https://www.alphavantage.co/support/#api-key
  Recommend use https://10minutemail.com/ to generate a temporary email to use
```

4.  Update user persona which can be found in `streamlit_app/data/user_persona.txt` using the user persona template which can be found in `streamlit_app/data/user_persona_template.txt`. Instructions have been provided - replace the curly brackets with relevant information.

5.  . [Optional] Update system persona, if needed, which can be found in `streamlit_app/data/system_persona.txt`.
6.  Run:

```
cd streamlit_app
streamlit run app.py
```

## Usage

```python
from dotenv import load_dotenv

load_dotenv()

from llama_index.packs.memary.agent.chat_agent import ChatAgent

system_persona_txt = "data/system_persona.txt"
user_persona_txt = "data/user_persona.txt"
past_chat_json = "data/past_chat.json"
memory_stream_json = "data/memory_stream.json"
entity_knowledge_store_json = "data/entity_knowledge_store.json"

chat_agent = ChatAgent(
    "Personal Agent",
    memory_stream_json,
    entity_knowledge_store_json,
    system_persona_txt,
    user_persona_txt,
    past_chat_json,
)
```

Pass in subset of `['search', 'vision', 'locate', 'stocks']` as `include_from_defaults` for different set of default tools upon initialization.

### Adding Custom Tools

```python
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


chat_agent.add_tool({"multiply": multiply})
```

More information about creating custom tools for the LlamaIndex ReAct Agent can be found [here](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/).

### Removing Tools

```python
chat_agent.remove_tool("multiply")
```
