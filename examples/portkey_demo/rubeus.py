from llama_index.llms.rubeus import Rubeus
from llama_index.llms.portkey_utils import LLMBase, ProviderTypes

client = Rubeus(
    api_key="",
    default_params={
        "messages": [
            {
                "role": "user",
                "content": "What are the top 10 happiest countries in the world?",
            }
        ],
        "max_tokens": 50,
    },
)

print("This is done..")
openai_llm = LLMBase(
    provider="openai",
    model="gpt-3.5-turbo",
    model_api_key="",
)
res = client.chat_completion.with_fallbacks(llms=[openai_llm])

print(res.json())
