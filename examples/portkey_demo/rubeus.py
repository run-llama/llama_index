from llama_index.llms.rubeus import Rubeus

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

res = client.chat_completion.create(
    provider="openai",
    model_api_key="",
    weight=1.0,
    model="gpt-3.5-turbo",
)

print(res.json())
